"""
Utility functions for Voronoi-based Pu-Cl cluster analysis.

This module provides:
- File system change logging system
- Basic OVITO pipeline utilities for trajectory loading

Functions are implemented/inspired by cluster_analysis modules.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

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
    relative_face_threshold: float = 0.0,
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
        relative_face_threshold: Minimum relative face area threshold (default: 0.0)
            Faces with area < relative_face_threshold * total_cell_area are filtered out atom-wise
            by OVITO during tessellation. This is more efficient than post-processing and ensures
            atom-wise thresholding (not pair-wise). If 0.0, all faces are included.
    
    Returns:
        DataCollection object with Voronoi analysis results
    
    References:
        voronoi/ovito/vornoi_ovito_cannon.ipynb
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    # Create VoronoiAnalysisModifier with relative_face_threshold
    voro = VoronoiAnalysisModifier(
        compute_indices=compute_indices,
        generate_bonds=generate_bonds,
        use_radii=use_radii,
        edge_threshold=edge_threshold,
        generate_polyhedra=generate_polyhedra,
        relative_face_threshold=relative_face_threshold,
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
    relative_face_threshold: float = 0.0,
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
        relative_face_threshold: Minimum relative face area threshold (default: 0.0)
            Faces with area < relative_face_threshold * total_cell_area are filtered out atom-wise
            by OVITO during tessellation. Use 0.0 for Phase 2/3 (all faces needed for coordination).
            Use non-zero value for Phase 4 (filtered faces for graph construction).
    
    Returns:
        Dictionary containing:
            - data: DataCollection object
            - bonds: Dictionary with pairs, face_areas, num_bonds (filtered if relative_face_threshold > 0)
            - particle_properties: Dictionary with atomic_volume, coordination, voronoi_index
            - face_areas: Array of face areas (from bonds or computed, filtered if relative_face_threshold > 0)
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
        relative_face_threshold=relative_face_threshold,
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


# ============================================================================
# Phase 3: Weighted Coordination Analysis
# ============================================================================

def compute_solid_angles(
    positions: np.ndarray,
    pairs: np.ndarray,
    face_areas: np.ndarray,
) -> np.ndarray:
    """
    Compute solid angles Ω = A / r² for each Voronoi face.
    
    This function implements Phase 3.2: Compute solid angles (Ω = A / r²) for each face.
    
    Args:
        positions: (N, 3) array of particle positions
        pairs: (M, 2) array of atom index pairs connected by Voronoi faces
        face_areas: (M,) array of face areas
    
    Returns:
        (M,) array of solid angles
    
    Formula:
        Ω_i = A_i / ||r_j - r_i||² for each pair (i, j)
    """
    if len(pairs) == 0:
        return np.array([], dtype=float)
    
    if len(face_areas) != len(pairs):
        raise ValueError(f"face_areas length ({len(face_areas)}) must match pairs length ({len(pairs)})")
    
    solid_angles = np.zeros(len(pairs), dtype=float)
    
    for idx, (i, j) in enumerate(pairs):
        if i >= len(positions) or j >= len(positions):
            solid_angles[idx] = 0.0
            continue
        
        # Compute distance between atoms
        r_vec = positions[j] - positions[i]
        distance_squared = np.dot(r_vec, r_vec)
        
        # Compute solid angle: Ω = A / r²
        if distance_squared > 0 and face_areas[idx] > 0:
            solid_angles[idx] = face_areas[idx] / distance_squared
        else:
            solid_angles[idx] = 0.0
    
    return solid_angles


def compute_topological_coordination(
    pairs: np.ndarray,
    species: np.ndarray,
    center_species: str = 'Pu',
    neighbor_species: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute topological coordination number (simple face count).
    
    This function counts the number of Voronoi faces (neighbors) for each center atom.
    Each Voronoi face corresponds to one nearest neighbor.
    
    Args:
        pairs: (M, 2) array of atom index pairs
        species: (N,) array of species names/identifiers
        center_species: Species to compute coordination for (default: 'Pu')
        neighbor_species: Optional species to filter neighbors by (None = all neighbors)
    
    Returns:
        Dictionary containing:
            - cn_values: (K,) array of integer coordination numbers for center_species atoms
            - atom_indices: (K,) array of atom indices for center_species atoms
    """
    if len(pairs) == 0:
        return {
            'cn_values': np.array([], dtype=int),
            'atom_indices': np.array([], dtype=int),
        }
    
    # Find all atoms of center_species
    center_mask = (species == center_species)
    center_indices = np.where(center_mask)[0]
    
    if len(center_indices) == 0:
        return {
            'cn_values': np.array([], dtype=int),
            'atom_indices': np.array([], dtype=int),
        }
    
    # Count neighbors for each center atom
    cn_values = np.zeros(len(center_indices), dtype=int)
    
    for center_idx, atom_idx in enumerate(center_indices):
        count = 0
        
        for i, j in pairs:
            # Check if this pair involves the center atom
            if i == atom_idx:
                neighbor_idx = j
            elif j == atom_idx:
                neighbor_idx = i
            else:
                continue
            
            # Filter by neighbor_species if specified
            if neighbor_species is not None:
                if neighbor_idx >= len(species) or species[neighbor_idx] != neighbor_species:
                    continue
            
            count += 1
        
        cn_values[center_idx] = count
    
    return {
        'cn_values': cn_values,
        'atom_indices': center_indices,
    }


def compute_area_weighted_coordination(
    pairs: np.ndarray,
    face_areas: np.ndarray,
    species: np.ndarray,
    center_species: str = 'Pu',
    neighbor_species: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute area-weighted coordination number CN_A = (Σ A_i)² / Σ A_i².
    
    This function implements Phase 3.3: Implement area-weighted coordination (CN_A).
    
    Args:
        pairs: (M, 2) array of atom index pairs
        face_areas: (M,) array of face areas
        species: (N,) array of species names/identifiers
        center_species: Species to compute coordination for (default: 'Pu')
        neighbor_species: Optional species to filter neighbors by (None = all neighbors)
    
    Returns:
        Dictionary containing:
            - cn_values: (K,) array of CN_A values for center_species atoms
            - neighbor_weights: Dictionary mapping atom_idx to array of neighbor weights
            - atom_indices: (K,) array of atom indices for center_species atoms
    
    Formula:
        CN_A = (Σ A_i)² / Σ A_i²
    """
    if len(pairs) == 0:
        return {
            'cn_values': np.array([], dtype=float),
            'neighbor_weights': {},
            'atom_indices': np.array([], dtype=int),
        }
    
    # Find all atoms of center_species
    center_mask = (species == center_species)
    center_indices = np.where(center_mask)[0]
    
    if len(center_indices) == 0:
        return {
            'cn_values': np.array([], dtype=float),
            'neighbor_weights': {},
            'atom_indices': np.array([], dtype=int),
        }
    
    # Build neighbor lists for each center atom
    cn_values = np.zeros(len(center_indices), dtype=float)
    neighbor_weights = {}
    
    for center_idx, atom_idx in enumerate(center_indices):
        # Find all pairs where this atom is the first or second element
        # and filter by neighbor_species if specified
        neighbor_areas = []
        
        for pair_idx, (i, j) in enumerate(pairs):
            # Check if this pair involves the center atom
            if i == atom_idx:
                neighbor_idx = j
            elif j == atom_idx:
                neighbor_idx = i
            else:
                continue
            
            # Filter by neighbor_species if specified
            if neighbor_species is not None:
                if neighbor_idx >= len(species) or species[neighbor_idx] != neighbor_species:
                    continue
            
            # Add face area
            if pair_idx < len(face_areas) and face_areas[pair_idx] > 0:
                neighbor_areas.append(face_areas[pair_idx])
        
        neighbor_areas = np.array(neighbor_areas)
        
        # Compute CN_A = (Σ A_i)² / Σ A_i²
        if len(neighbor_areas) > 0:
            sum_areas = np.sum(neighbor_areas)
            sum_areas_squared = np.sum(neighbor_areas ** 2)
            
            if sum_areas_squared > 0:
                cn_a = (sum_areas ** 2) / sum_areas_squared
            else:
                cn_a = 0.0
            
            # Store normalized weights
            if sum_areas > 0:
                normalized_weights = neighbor_areas / sum_areas
            else:
                normalized_weights = neighbor_areas
            neighbor_weights[atom_idx] = normalized_weights
        else:
            cn_a = 0.0
            neighbor_weights[atom_idx] = np.array([], dtype=float)
        
        cn_values[center_idx] = cn_a
    
    return {
        'cn_values': cn_values,
        'neighbor_weights': neighbor_weights,
        'atom_indices': center_indices,
    }


def compute_solid_angle_weighted_coordination(
    positions: np.ndarray,
    pairs: np.ndarray,
    face_areas: np.ndarray,
    species: np.ndarray,
    center_species: str = 'Pu',
    neighbor_species: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute solid-angle-weighted coordination number CN_Ω = (Σ Ω_i)² / Σ Ω_i².
    
    This function implements Phase 3.4: Implement solid-angle-weighted coordination (CN_Ω).
    
    Args:
        positions: (N, 3) array of particle positions
        pairs: (M, 2) array of atom index pairs
        face_areas: (M,) array of face areas
        species: (N,) array of species names/identifiers
        center_species: Species to compute coordination for (default: 'Pu')
        neighbor_species: Optional species to filter neighbors by (None = all neighbors)
    
    Returns:
        Dictionary containing:
            - cn_values: (K,) array of CN_Ω values for center_species atoms
            - neighbor_weights: Dictionary mapping atom_idx to array of neighbor solid angle weights
            - atom_indices: (K,) array of atom indices for center_species atoms
    
    Formula:
        CN_Ω = (Σ Ω_i)² / Σ Ω_i² where Ω_i = A_i / r_i²
    """
    if len(pairs) == 0:
        return {
            'cn_values': np.array([], dtype=float),
            'neighbor_weights': {},
            'atom_indices': np.array([], dtype=int),
        }
    
    # Compute solid angles first
    solid_angles = compute_solid_angles(positions, pairs, face_areas)
    
    # Find all atoms of center_species
    center_mask = (species == center_species)
    center_indices = np.where(center_mask)[0]
    
    if len(center_indices) == 0:
        return {
            'cn_values': np.array([], dtype=float),
            'neighbor_weights': {},
            'atom_indices': np.array([], dtype=int),
        }
    
    # Build neighbor lists for each center atom
    cn_values = np.zeros(len(center_indices), dtype=float)
    neighbor_weights = {}
    
    for center_idx, atom_idx in enumerate(center_indices):
        # Find all pairs where this atom is the first or second element
        # and filter by neighbor_species if specified
        neighbor_solid_angles = []
        
        for pair_idx, (i, j) in enumerate(pairs):
            # Check if this pair involves the center atom
            if i == atom_idx:
                neighbor_idx = j
            elif j == atom_idx:
                neighbor_idx = i
            else:
                continue
            
            # Filter by neighbor_species if specified
            if neighbor_species is not None:
                if neighbor_idx >= len(species) or species[neighbor_idx] != neighbor_species:
                    continue
            
            # Add solid angle
            if pair_idx < len(solid_angles) and solid_angles[pair_idx] > 0:
                neighbor_solid_angles.append(solid_angles[pair_idx])
        
        neighbor_solid_angles = np.array(neighbor_solid_angles)
        
        # Compute CN_Ω = (Σ Ω_i)² / Σ Ω_i²
        if len(neighbor_solid_angles) > 0:
            sum_omega = np.sum(neighbor_solid_angles)
            sum_omega_squared = np.sum(neighbor_solid_angles ** 2)
            
            if sum_omega_squared > 0:
                cn_omega = (sum_omega ** 2) / sum_omega_squared
            else:
                cn_omega = 0.0
            
            # Store normalized weights
            if sum_omega > 0:
                normalized_weights = neighbor_solid_angles / sum_omega
            else:
                normalized_weights = neighbor_solid_angles
            neighbor_weights[atom_idx] = normalized_weights
        else:
            cn_omega = 0.0
            neighbor_weights[atom_idx] = np.array([], dtype=float)
        
        cn_values[center_idx] = cn_omega
    
    return {
        'cn_values': cn_values,
        'neighbor_weights': neighbor_weights,
        'atom_indices': center_indices,
    }


def build_coordination_histograms(
    pairs: np.ndarray,
    species: np.ndarray,
    center_species: str = 'Pu',
    bins: Optional[np.ndarray] = None,
    coordination_type: str = 'topological',
    positions: Optional[np.ndarray] = None,
    face_areas: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build coordination histograms for Pu with different neighbor types.
    
    This function implements Phase 3.5: Build coordination histograms for Pu with neighbors:
    - Pu-Cl coordination histogram
    - Pu-Na coordination histogram
    - Pu-Pu coordination histogram
    - Pu-Any (total) coordination histogram
    
    Supports different coordination types:
    - 'topological': Simple neighbor count (number of faces)
    - 'area': Area-weighted coordination (CN_A)
    - 'solid_angle': Solid-angle-weighted coordination (CN_Ω)
    
    Args:
        pairs: (M, 2) array of atom index pairs
        species: (N,) array of species names/identifiers
        center_species: Species to compute coordination for (default: 'Pu')
        bins: Optional bin edges for histograms (None = auto)
        coordination_type: Type of coordination to use ('topological', 'area', 'solid_angle')
        positions: (N, 3) array of particle positions (required for 'area' and 'solid_angle')
        face_areas: (M,) array of face areas (required for 'area' and 'solid_angle')
    
    Returns:
        Dictionary containing:
            - pu_cl: Histogram data for Pu-Cl coordination
            - pu_na: Histogram data for Pu-Na coordination
            - pu_pu: Histogram data for Pu-Pu coordination
            - pu_any: Histogram data for Pu-Any (total) coordination
            - bins: Bin edges used (same for all histograms)
            - coordination_type: Type of coordination used
    """
    if len(pairs) == 0:
        empty_hist = {'counts': np.array([], dtype=int), 'bin_edges': np.array([], dtype=float)}
        return {
            'pu_cl': empty_hist,
            'pu_na': empty_hist,
            'pu_pu': empty_hist,
            'pu_any': empty_hist,
            'bins': np.array([], dtype=float),
            'coordination_type': coordination_type,
        }
    
    # Validate coordination_type
    if coordination_type not in ['topological', 'area', 'solid_angle']:
        raise ValueError(f"coordination_type must be 'topological', 'area', or 'solid_angle', got '{coordination_type}'")
    
    # Validate required parameters for weighted coordination
    if coordination_type in ['area', 'solid_angle']:
        if positions is None:
            raise ValueError(f"positions required for coordination_type='{coordination_type}'")
        if face_areas is None:
            raise ValueError(f"face_areas required for coordination_type='{coordination_type}'")
        if len(face_areas) != len(pairs):
            raise ValueError(f"face_areas length ({len(face_areas)}) must match pairs length ({len(pairs)})")
    
    # Find all atoms of center_species
    center_mask = (species == center_species)
    center_indices = np.where(center_mask)[0]
    
    if len(center_indices) == 0:
        empty_hist = {'counts': np.array([], dtype=int), 'bin_edges': np.array([], dtype=float)}
        return {
            'pu_cl': empty_hist,
            'pu_na': empty_hist,
            'pu_pu': empty_hist,
            'pu_any': empty_hist,
            'bins': np.array([], dtype=float),
            'coordination_type': coordination_type,
        }
    
    # Compute coordination values based on type
    if coordination_type == 'topological':
        # Use simple neighbor counts
        cn_cl = compute_topological_coordination(pairs, species, center_species, neighbor_species='Cl')
        cn_na = compute_topological_coordination(pairs, species, center_species, neighbor_species='Na')
        cn_pu = compute_topological_coordination(pairs, species, center_species, neighbor_species='Pu')
        cn_any = compute_topological_coordination(pairs, species, center_species, neighbor_species=None)
        
        coordination_pu_cl = cn_cl['cn_values'].astype(float)
        coordination_pu_na = cn_na['cn_values'].astype(float)
        coordination_pu_pu = cn_pu['cn_values'].astype(float)
        coordination_pu_any = cn_any['cn_values'].astype(float)
        
    elif coordination_type == 'area':
        # Use area-weighted coordination
        cn_cl = compute_area_weighted_coordination(pairs, face_areas, species, center_species, neighbor_species='Cl')
        cn_na = compute_area_weighted_coordination(pairs, face_areas, species, center_species, neighbor_species='Na')
        cn_pu = compute_area_weighted_coordination(pairs, face_areas, species, center_species, neighbor_species='Pu')
        cn_any = compute_area_weighted_coordination(pairs, face_areas, species, center_species, neighbor_species=None)
        
        coordination_pu_cl = cn_cl['cn_values']
        coordination_pu_na = cn_na['cn_values']
        coordination_pu_pu = cn_pu['cn_values']
        coordination_pu_any = cn_any['cn_values']
        
    elif coordination_type == 'solid_angle':
        # Use solid-angle-weighted coordination
        cn_cl = compute_solid_angle_weighted_coordination(positions, pairs, face_areas, species, center_species, neighbor_species='Cl')
        cn_na = compute_solid_angle_weighted_coordination(positions, pairs, face_areas, species, center_species, neighbor_species='Na')
        cn_pu = compute_solid_angle_weighted_coordination(positions, pairs, face_areas, species, center_species, neighbor_species='Pu')
        cn_any = compute_solid_angle_weighted_coordination(positions, pairs, face_areas, species, center_species, neighbor_species=None)
        
        coordination_pu_cl = cn_cl['cn_values']
        coordination_pu_na = cn_na['cn_values']
        coordination_pu_pu = cn_pu['cn_values']
        coordination_pu_any = cn_any['cn_values']
    
    # Determine bin edges
    if bins is None:
        # Auto-determine bins based on data range
        all_values = np.concatenate([
            coordination_pu_cl, coordination_pu_na, 
            coordination_pu_pu, coordination_pu_any
        ])
        if len(all_values) > 0:
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            # Use appropriate binning based on coordination type
            if coordination_type == 'topological':
                # Integer bins for topological
                bins = np.arange(int(min_val), int(max_val) + 2) - 0.5
            else:
                # Continuous bins for weighted coordination
                num_bins = min(50, max(20, int((max_val - min_val) * 2)))
                bins = np.linspace(min_val, max_val, num_bins + 1)
        else:
            bins = np.array([0, 1])
    
    # Compute histograms
    counts_pu_cl, _ = np.histogram(coordination_pu_cl, bins=bins)
    counts_pu_na, _ = np.histogram(coordination_pu_na, bins=bins)
    counts_pu_pu, _ = np.histogram(coordination_pu_pu, bins=bins)
    counts_pu_any, _ = np.histogram(coordination_pu_any, bins=bins)
    
    return {
        'pu_cl': {'counts': counts_pu_cl, 'bin_edges': bins, 'values': coordination_pu_cl},
        'pu_na': {'counts': counts_pu_na, 'bin_edges': bins, 'values': coordination_pu_na},
        'pu_pu': {'counts': counts_pu_pu, 'bin_edges': bins, 'values': coordination_pu_pu},
        'pu_any': {'counts': counts_pu_any, 'bin_edges': bins, 'values': coordination_pu_any},
        'bins': bins,
        'coordination_type': coordination_type,
    }


def compute_weighted_coordination_analysis(
    data: DataCollection,
    bonds: Dict[str, Any],
    particle_properties: Dict[str, Any],
    face_areas: np.ndarray,
    center_species: str = 'Pu',
) -> Dict[str, Any]:
    """
    Main wrapper function for weighted coordination analysis.
    
    This function implements Phase 3.6: Main wrapper function that combines all Phase 3 steps.
    
    Args:
        data: DataCollection from apply_voronoi_analysis()
        bonds: Dictionary from extract_voronoi_bonds() with pairs
        particle_properties: Dictionary from extract_voronoi_particle_properties()
        face_areas: (M,) array of face areas
        center_species: Species to analyze (default: 'Pu')
    
    Returns:
        Dictionary containing:
            - cn_topological_all: Topological CN values for all neighbors
            - cn_topological_cl: Topological CN values for Cl neighbors
            - cn_topological_na: Topological CN values for Na neighbors
            - cn_topological_pu: Topological CN values for Pu neighbors
            - cn_a_all: CN_A values for all neighbors
            - cn_a_cl: CN_A values for Cl neighbors
            - cn_a_na: CN_A values for Na neighbors
            - cn_a_pu: CN_A values for Pu neighbors
            - cn_omega_all: CN_Ω values for all neighbors
            - cn_omega_cl: CN_Ω values for Cl neighbors
            - cn_omega_na: CN_Ω values for Na neighbors
            - cn_omega_pu: CN_Ω values for Pu neighbors
            - histograms: Dictionary with coordination histograms (topological by default)
            - positions: (N, 3) array of positions
            - species: (N,) array of species
    """
    if not OVITO_AVAILABLE:
        raise ImportError("OVITO is required")
    
    # Extract positions and species
    particle_data = extract_particle_properties(data)
    positions = particle_data['positions']
    species_array = particle_data['species']
    
    if positions is None or species_array is None:
        raise ValueError("Could not extract positions or species from data")
    
    pairs = bonds['pairs']
    
    # Compute topological coordination for different neighbor types
    cn_topological_all = compute_topological_coordination(
        pairs, species_array, center_species, neighbor_species=None
    )
    cn_topological_cl = compute_topological_coordination(
        pairs, species_array, center_species, neighbor_species='Cl'
    )
    cn_topological_na = compute_topological_coordination(
        pairs, species_array, center_species, neighbor_species='Na'
    )
    cn_topological_pu = compute_topological_coordination(
        pairs, species_array, center_species, neighbor_species='Pu'
    )
    
    # Compute CN_A for different neighbor types
    cn_a_all = compute_area_weighted_coordination(
        pairs, face_areas, species_array, center_species, neighbor_species=None
    )
    cn_a_cl = compute_area_weighted_coordination(
        pairs, face_areas, species_array, center_species, neighbor_species='Cl'
    )
    cn_a_na = compute_area_weighted_coordination(
        pairs, face_areas, species_array, center_species, neighbor_species='Na'
    )
    cn_a_pu = compute_area_weighted_coordination(
        pairs, face_areas, species_array, center_species, neighbor_species='Pu'
    )
    
    # Compute CN_Ω for different neighbor types
    cn_omega_all = compute_solid_angle_weighted_coordination(
        positions, pairs, face_areas, species_array, center_species, neighbor_species=None
    )
    cn_omega_cl = compute_solid_angle_weighted_coordination(
        positions, pairs, face_areas, species_array, center_species, neighbor_species='Cl'
    )
    cn_omega_na = compute_solid_angle_weighted_coordination(
        positions, pairs, face_areas, species_array, center_species, neighbor_species='Na'
    )
    cn_omega_pu = compute_solid_angle_weighted_coordination(
        positions, pairs, face_areas, species_array, center_species, neighbor_species='Pu'
    )
    
    # Build coordination histograms (default: topological)
    histograms = build_coordination_histograms(
        pairs, species_array, center_species, bins=None
    )
    
    return {
        'cn_topological_all': cn_topological_all,
        'cn_topological_cl': cn_topological_cl,
        'cn_topological_na': cn_topological_na,
        'cn_topological_pu': cn_topological_pu,
        'cn_a_all': cn_a_all,
        'cn_a_cl': cn_a_cl,
        'cn_a_na': cn_a_na,
        'cn_a_pu': cn_a_pu,
        'cn_omega_all': cn_omega_all,
        'cn_omega_cl': cn_omega_cl,
        'cn_omega_na': cn_omega_na,
        'cn_omega_pu': cn_omega_pu,
        'histograms': histograms,
        'positions': positions,
        'species': species_array,
    }


# ============================================================================
# Phase 4: Area Thresholding and Shared Anion Graph
# ============================================================================

def build_shared_anion_graph_from_voronoi(
    pairs: np.ndarray,
    face_areas: np.ndarray,
    positions: np.ndarray,
    species: np.ndarray,
    cation: str = 'Pu',
    anion: str = 'Cl',
) -> Any:
    """
    Build a shared anion graph from Voronoi tessellation faces.
    
    This function implements Phase 4.2: Build shared anion graph from Voronoi tessellation.
    
    Creates a networkx Graph with:
    - Nodes: All Pu and Cl atoms (with attributes: position, species, index)
    - Edges: Connections via shared Voronoi faces (with attribute: area)
    
    Only includes edges between Pu-Cl (cation-anion) pairs.
    
    Args:
        pairs: (M, 2) array of atom index pairs (can be pre-filtered or raw)
        face_areas: (M,) array of face areas corresponding to pairs
        positions: (N, 3) array of atom positions
        species: (N,) array of species names/identifiers
        cation: Species to use as cation (default: 'Pu')
        anion: Species to use as anion (default: 'Cl')
    
    Returns:
        networkx Graph with Pu and Cl nodes and their connections
    
    Raises:
        ImportError: If networkx is not available
    
    Examples:
        >>> pairs = np.array([[0, 1], [1, 2]])
        >>> areas = np.array([10.0, 5.0])
        >>> positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> species = np.array(['Pu', 'Cl', 'Pu'])
        >>> graph = build_shared_anion_graph_from_voronoi(pairs, areas, positions, species)
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for build_shared_anion_graph_from_voronoi")
    
    if len(pairs) == 0:
        # Return empty graph but still add nodes for Pu and Cl atoms
        G = nx.Graph()
        # Add nodes for all Pu and Cl atoms
        for i in range(len(species)):
            if species[i] == cation or species[i] == anion:
                G.add_node(
                    int(i),
                    position=np.asarray(positions[i]),
                    species=str(species[i]),
                    index=int(i)
                )
        return G
    
    if len(face_areas) != len(pairs):
        raise ValueError(f"face_areas length ({len(face_areas)}) must match pairs length ({len(pairs)})")
    
    if len(positions) != len(species):
        raise ValueError(f"positions length ({len(positions)}) must match species length ({len(species)})")
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes for all Pu and Cl atoms
    for i in range(len(species)):
        if species[i] == cation or species[i] == anion:
            G.add_node(
                int(i),
                position=np.asarray(positions[i]),
                species=str(species[i]),
                index=int(i)
            )
    
    # Add edges for pairs involving Pu or Cl atoms
    for idx, (i, j) in enumerate(pairs):
        i_idx = int(i)
        j_idx = int(j)
        
        # Check if both atoms are within bounds
        if i_idx >= len(species) or j_idx >= len(species):
            continue
        
        # Only add edges if both atoms are Pu or Cl
        if (species[i_idx] == cation and species[j_idx] == anion) or \
           (species[i_idx] == anion and species[j_idx] == cation):
            area = face_areas[idx] if idx < len(face_areas) else 0.0
            G.add_edge(
                i_idx,
                j_idx,
                area=float(area),
                species_pair=f"{species[i_idx]}-{species[j_idx]}"
            )
    
    return G


def assign_cluster_ids_from_graph(
    graph: Any,
    num_atoms: int,
    species: np.ndarray,
    cation: str = 'Pu',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign cluster IDs to all atoms based on connected components in the graph.
    
    This function implements Phase 4.3: Assign cluster IDs from graph.
    
    Uses networkx connected_components to find all connected components.
    Clusters are sorted by number of cation atoms (descending) before assigning IDs,
    so cluster_id=0 corresponds to the cluster with the most cation atoms.
    All atoms (Pu and Cl) in the same component get the same cluster ID.
    Na atoms and unconnected atoms get cluster_id = -1.
    
    Args:
        graph: networkx Graph from build_shared_anion_graph_from_voronoi
        num_atoms: Total number of atoms in system
        species: (N,) array of species names (required for sorting by cation count)
        cation: Species to use as cation for sorting (default: 'Pu')
    
    Returns:
        Tuple containing:
            - cluster_ids: (N,) array of cluster IDs (-1 for unclustered)
            - cluster_sizes: Array of cluster sizes (sorted by cation count, descending)
    
    Raises:
        ImportError: If networkx is not available
    
    Examples:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(1, 2)
        >>> species = np.array(['Pu', 'Cl', 'Pu', 'Na', 'Na'])
        >>> cluster_ids, cluster_sizes = assign_cluster_ids_from_graph(G, num_atoms=5, species=species)
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for assign_cluster_ids_from_graph")
    
    if len(species) != num_atoms:
        raise ValueError(f"species length ({len(species)}) must match num_atoms ({num_atoms})")
    
    # Initialize cluster_ids array with -1 (unclustered)
    cluster_ids = -np.ones(num_atoms, dtype=int)
    
    # Find connected components
    components = list(nx.connected_components(graph))
    
    # Sort components by number of cation atoms (descending)
    def count_cations(comp):
        return sum(1 for idx in comp if int(idx) < len(species) and species[int(idx)] == cation)
    
    sorted_components = sorted(components, key=count_cations, reverse=True)
    
    # Assign cluster IDs based on sorted order
    for cluster_id, component in enumerate(sorted_components):
        for atom_idx in component:
            atom_idx_int = int(atom_idx)
            if atom_idx_int < num_atoms:
                cluster_ids[atom_idx_int] = cluster_id
    
    # Compute cluster sizes (matching sorted order)
    cluster_sizes = np.array([len(comp) for comp in sorted_components], dtype=int)
    
    return cluster_ids, cluster_sizes
