"""
Utility functions for Voronoi-based Pu-Cl cluster analysis.

This module provides:
- File system change logging system
- Basic OVITO pipeline utilities for trajectory loading

Functions are implemented/inspired by cluster_analysis modules.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np


from ovito.io import import_file
from ovito.pipeline import Pipeline
from ovito.data import DataCollection
from ovito.modifiers import VoronoiAnalysisModifier
OVITO_AVAILABLE = True



# Path to log file (relative to project root)
_LOG_FILE = Path(__file__).parent / "filesystem_changes.log"


# ============================================================================
# File System Logging System
# ============================================================================

def log_filesystem_change(action: str, file_path: str, details: Optional[str] = None) -> None:
    """
    Log a file system change to filesystem_changes.log.
    
    Args:
        action: Action type (CREATE, MODIFY, DELETE, MKDIR, RMDIR, MOVE, RENAME)
        file_path: Path to the file or directory
        details: Optional additional details (e.g., size, lines changed, reason)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Normalize path for logging
    log_path = str(file_path)
    
    # Format log entry
    if details:
        log_entry = f"[{timestamp}] {action} {log_path} [{details}]\n"
    else:
        log_entry = f"[{timestamp}] {action} {log_path}\n"
    
    # Append to log file
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


# ============================================================================
# OVITO Pipeline Utilities
# ============================================================================

def get_pipeline_info(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Extract basic information from an OVITO pipeline.
    
    Args:
        pipeline: OVITO Pipeline object
    
    Returns:
        Dictionary containing:
            - num_frames: Number of frames in trajectory
            - num_particles: Number of particles (from first frame)
            - species: List of unique species
            - cell_info: Simulation cell information
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    # Get first frame to extract basic info
    data = pipeline.compute(0)
    particles = data.particles
    
    # Get number of frames
    num_frames = pipeline.source.num_frames if hasattr(pipeline.source, 'num_frames') else 1
    
    # Get number of particles
    num_particles = particles.count if hasattr(particles, 'count') else len(particles.position)
    
    # Extract species
    species = []
    if hasattr(particles, 'elements') and particles.elements:
        species = list(set([el.symbol for el in particles.elements]))
    elif "Particle Type" in particles:
        # Try to get from particle types using type_by_id
        try:
            types_prop = particles["Particle Type"]
            types = particles.particle_types
            if types is not None:
                # Get unique type IDs and convert to names
                # Property objects are iterable, so we can iterate directly
                unique_type_ids = set(types_prop)
                species = list(set([types.type_by_id(t).name for t in unique_type_ids]))
        except:
            pass
    
    # Get cell information
    cell_info = {}
    if hasattr(data, 'cell') and data.cell:
        cell = data.cell
        cell_info = {
            'matrix': np.array(cell.matrix) if hasattr(cell, 'matrix') else None,
            'pbc': tuple(cell.pbc) if hasattr(cell, 'pbc') else (True, True, True),
            'volume': float(cell.volume) if hasattr(cell, 'volume') else None,
        }
    
    return {
        'num_frames': num_frames,
        'num_particles': num_particles,
        'species': species,
        'cell_info': cell_info,
    }


def extract_particle_properties(data: DataCollection) -> Dict[str, np.ndarray]:
    """
    Extract particle properties from OVITO DataCollection.
    
    Args:
        data: OVITO DataCollection object
    
    Returns:
        Dictionary containing:
            - positions: (N, 3) array of particle positions
            - species: (N,) array of species/type names
            - cell_matrix: (3, 4) simulation cell matrix
            - pbc: Tuple of periodic boundary conditions
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    particles = data.particles
    
    # Extract positions
    positions = np.array(particles.position) if hasattr(particles, 'position') else None
    
    # Extract species
    species = None
    if hasattr(particles, 'elements') and particles.elements:
        species = np.array([el.symbol for el in particles.elements])
    elif "Particle Type" in particles:
        try:
            types_prop = particles["Particle Type"]
            types = particles.particle_types
            if types is not None:
                # Property objects are iterable, so we can iterate directly
                # Convert each type ID to name using type_by_id
                species = np.array([types.type_by_id(t).name for t in types_prop])
        except:
            pass
    
    # Extract cell information
    cell_matrix = None
    pbc = (True, True, True)
    
    if hasattr(data, 'cell') and data.cell:
        cell = data.cell
        if hasattr(cell, 'matrix'):
            cell_matrix = np.array(cell.matrix)
        if hasattr(cell, 'pbc'):
            pbc = tuple(cell.pbc)
    
    return {
        'positions': positions,
        'species': species,
        'cell_matrix': cell_matrix,
        'pbc': pbc,
    }


# ============================================================================
# Phase 2: Voronoi Tessellation
# ============================================================================

def apply_voronoi_analysis(
    pipeline: Pipeline,
    frame: int = 0,
    generate_bonds: bool = True,
    compute_indices: bool = True,
    use_radii: bool = False,
    edge_threshold: float = 0.0,
    generate_polyhedra: bool = True,
) -> DataCollection:
    """
    Apply VoronoiAnalysisModifier to a pipeline and return the computed DataCollection.
    
    This function implements Phase 2.1: Apply VoronoiAnalysisModifier with generate_bonds=True.
    
    Args:
        pipeline: OVITO Pipeline object
        frame: Frame index to compute (default: 0)
        generate_bonds: Whether to generate bonds representing Voronoi faces (default: True)
        compute_indices: Whether to compute Voronoi indices (default: True)
        use_radii: Whether to use atomic radii for weighted Voronoi tessellation (default: False)
        edge_threshold: Minimum edge length threshold (default: 0.0)
        generate_polyhedra: Whether to output Voronoi cells as polyhedral SurfaceMesh (default: True)
            Enables extraction of face areas from the surface mesh.
    
    Returns:
        DataCollection object with Voronoi analysis results
    
    References:
        voronoi/ovito/vornoi_ovito_cannon.ipynb
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    # Create VoronoiAnalysisModifier
    voro = VoronoiAnalysisModifier(
        compute_indices=compute_indices,
        generate_bonds=generate_bonds,
        use_radii=use_radii,
        edge_threshold=edge_threshold,
        generate_polyhedra=generate_polyhedra,
    )
    
    # Add modifier to pipeline (temporarily)
    pipeline.modifiers.append(voro)
    
    try:
        # Compute frame with Voronoi analysis
        data = pipeline.compute(frame)
        return data
    finally:
        # Remove modifier after computation
        try:
            pipeline.modifiers.remove(voro)
        except Exception:
            pass


def extract_voronoi_bonds(data: DataCollection) -> Dict[str, Any]:
    """
    Extract Voronoi bonds (face connections) from DataCollection.
    
    This function implements Phase 2.3: Access Bonds object with topology (Voronoi face connections).
    
    Args:
        data: DataCollection from apply_voronoi_analysis()
    
    Returns:
        Dictionary containing:
            - pairs: (N, 2) array of atom index pairs connected by Voronoi faces
            - face_areas: (N,) array of face areas (extracted from surface mesh)
            - num_bonds: Number of Voronoi bonds/faces
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    particles = getattr(data, "particles", None)
    bonds = getattr(particles, "bonds", None) if particles is not None else None
    
    if bonds is None or len(bonds) == 0:
        return {
            'pairs': np.empty((0, 2), dtype=int),
            'face_areas': np.empty((0,), dtype=float),
            'num_bonds': 0,
        }
    
    # Extract topology (pairs)
    pairs = None
    if "Topology" in bonds:
        pairs = np.array(bonds["Topology"].array, dtype=int)
        if pairs.ndim == 1:
            pairs = pairs.reshape(-1, 2)
    elif hasattr(bonds, "topology"):
        pairs = np.array(bonds.topology, dtype=int)
    elif hasattr(bonds, "pairs"):
        pairs = np.array(bonds.pairs, dtype=int)
    else:
        pairs = np.empty((0, 2), dtype=int)
    
    # Extract face areas from surface mesh
    # Face areas are stored in the SurfaceMesh, not directly in bonds
    face_areas = np.zeros(len(pairs), dtype=float)
    
    # Try to access face areas from the Voronoi surface mesh
    if hasattr(data, 'surfaces') and 'voronoi-polyhedra' in data.surfaces:
        try:
            voronoi_mesh = data.surfaces['voronoi-polyhedra']
            if hasattr(voronoi_mesh, 'faces') and 'Area' in voronoi_mesh.faces:
                mesh_areas = np.array(voronoi_mesh.faces['Area'].array, dtype=float)
                
                # Map face areas to bond pairs
                # Each face in the mesh corresponds to a bond between two particles
                # We need to match faces to bonds using vertex/particle indices
                if hasattr(voronoi_mesh.faces, 'vertex_indices') or 'Vertex Indices' in voronoi_mesh.faces:
                    # Get face vertex indices (which correspond to particle indices)
                    if 'Vertex Indices' in voronoi_mesh.faces:
                        face_vertices = np.array(voronoi_mesh.faces['Vertex Indices'].array, dtype=int)
                    else:
                        face_vertices = np.array(voronoi_mesh.faces.vertex_indices, dtype=int)
                    
                    # Create a mapping from bond pairs to face areas
                    # Each face connects two particles, so we match by particle pair
                    face_areas_dict = {}
                    for i, face_verts in enumerate(face_vertices):
                        if len(face_verts) >= 2:
                            # Face vertices are particle indices
                            # Sort to make pairs consistent
                            p1, p2 = sorted(face_verts[:2])
                            face_areas_dict[(p1, p2)] = mesh_areas[i]
                    
                    # Map bond pairs to face areas
                    for idx, (p1, p2) in enumerate(pairs):
                        # Try both orderings
                        if (p1, p2) in face_areas_dict:
                            face_areas[idx] = face_areas_dict[(p1, p2)]
                        elif (p2, p1) in face_areas_dict:
                            face_areas[idx] = face_areas_dict[(p2, p1)]
                else:
                    # Fallback: assume faces match bonds in order
                    if len(mesh_areas) == len(pairs):
                        face_areas = mesh_areas
                    elif len(mesh_areas) > 0:
                        face_areas = mesh_areas[:len(pairs)]
        except Exception as e:
            # If surface mesh access fails, face areas remain zeros
            # This is expected if surface mesh is not available
            pass
    
    # Fallback: try to get face areas from bonds directly (if available)
    if np.all(face_areas == 0):
        if "Face Area" in bonds:
            face_areas = np.array(bonds["Face Area"].array, dtype=float)
        elif "Surface Area" in bonds:
            face_areas = np.array(bonds["Surface Area"].array, dtype=float)
        elif hasattr(bonds, "face_areas"):
            face_areas = np.array(bonds.face_areas, dtype=float)
    
    return {
        'pairs': pairs,
        'face_areas': face_areas,
        'num_bonds': len(pairs),
    }


def extract_voronoi_particle_properties(data: DataCollection) -> Dict[str, np.ndarray]:
    """
    Extract Voronoi particle properties: Atomic Volume, Coordination, Voronoi Index.
    
    This function implements Phase 2.4: Extract particle properties from DataCollection.
    
    Args:
        data: DataCollection from apply_voronoi_analysis()
    
    Returns:
        Dictionary containing:
            - atomic_volume: (N,) array of atomic volumes (Voronoi cell volumes)
            - coordination: (N,) array of coordination numbers
            - voronoi_index: (N, M) array of Voronoi indices (if computed)
            - available_properties: List of property names that were successfully extracted
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    particles = data.particles
    num_particles = particles.count if hasattr(particles, 'count') else len(particles.position)
    
    result = {
        'atomic_volume': None,
        'coordination': None,
        'voronoi_index': None,
        'available_properties': [],
    }
    
    # Extract Atomic Volume
    if "Atomic Volume" in particles:
        result['atomic_volume'] = np.array(particles["Atomic Volume"].array, dtype=float)
        result['available_properties'].append('Atomic Volume')
    elif hasattr(particles, 'atomic_volume'):
        result['atomic_volume'] = np.array(particles.atomic_volume, dtype=float)
        result['available_properties'].append('Atomic Volume')
    
    # Extract Coordination Number
    if "Coordination" in particles:
        result['coordination'] = np.array(particles["Coordination"].array, dtype=int)
        result['available_properties'].append('Coordination')
    elif hasattr(particles, 'coordination'):
        result['coordination'] = np.array(particles.coordination, dtype=int)
        result['available_properties'].append('Coordination')
    
    # Extract Voronoi Index (if compute_indices=True was used)
    if "Voronoi Index" in particles:
        voronoi_index_prop = particles["Voronoi Index"]
        # Voronoi Index is typically a 2D array (N, M) where M is the number of face types
        if hasattr(voronoi_index_prop, 'array'):
            result['voronoi_index'] = np.array(voronoi_index_prop.array, dtype=int)
            result['available_properties'].append('Voronoi Index')
    elif hasattr(particles, 'voronoi_index'):
        result['voronoi_index'] = np.array(particles.voronoi_index, dtype=int)
        result['available_properties'].append('Voronoi Index')
    
    return result


def compute_voronoi_face_areas(
    data: DataCollection,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    Compute face areas from Voronoi surface mesh (optional).
    
    This function implements Phase 2.5: Optionally compute surface mesh to extract face areas.
    
    Note: This is a placeholder for future implementation. Currently, face areas
    are extracted directly from bonds if available, otherwise set to zero.
    
    Args:
        data: DataCollection from apply_voronoi_analysis()
        pairs: (N, 2) array of atom index pairs
    
    Returns:
        (N,) array of face areas
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    # Try to extract face areas from bonds first
    bonds_info = extract_voronoi_bonds(data)
    face_areas = bonds_info['face_areas']
    
    # If face areas are all zeros, they were not computed
    # In the future, this could compute surface mesh to get actual areas
    # For now, return the extracted areas (which may be zeros)
    return face_areas


def perform_voronoi_tessellation(
    pipeline: Pipeline,
    frame: int = 0,
    use_radii: bool = False,
    edge_threshold: float = 0.0,
    compute_surface_mesh: bool = False,
) -> Dict[str, Any]:
    """
    Complete Voronoi tessellation analysis for a pipeline frame.
    
    This function combines all Phase 2 steps:
    1. Apply VoronoiAnalysisModifier
    2. Extract DataCollection
    3. Extract bonds topology
    4. Extract particle properties
    5. Optionally compute surface mesh face areas
    
    Args:
        pipeline: OVITO Pipeline object
        frame: Frame index to analyze (default: 0)
        use_radii: Whether to use atomic radii (default: False)
        edge_threshold: Minimum edge length threshold (default: 0.0)
        compute_surface_mesh: Whether to compute surface mesh for face areas (default: False)
            When True, enables generate_polyhedra to create SurfaceMesh with face area properties.
            When False, generate_polyhedra defaults to True for face area extraction.
    
    Returns:
        Dictionary containing:
            - data: DataCollection object
            - bonds: Dictionary with pairs, face_areas, num_bonds
            - particle_properties: Dictionary with atomic_volume, coordination, voronoi_index
            - face_areas: Array of face areas (from bonds or computed)
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    # Step 1: Apply VoronoiAnalysisModifier
    # Enable generate_polyhedra to create SurfaceMesh with face area properties
    data = apply_voronoi_analysis(
        pipeline=pipeline,
        frame=frame,
        generate_bonds=True,
        compute_indices=True,
        use_radii=use_radii,
        edge_threshold=edge_threshold,
        generate_polyhedra=True,
    )
    
    # Step 2: Extract bonds topology
    bonds = extract_voronoi_bonds(data)
    
    # Step 3: Extract particle properties
    particle_properties = extract_voronoi_particle_properties(data)
    
    # Step 4: Extract/compute face areas
    if compute_surface_mesh:
        # For now, use extracted areas (future: compute from surface mesh)
        face_areas = compute_voronoi_face_areas(data, bonds['pairs'])
    else:
        face_areas = bonds['face_areas']
    
    return {
        'data': data,
        'bonds': bonds,
        'particle_properties': particle_properties,
        'face_areas': face_areas,
    }

