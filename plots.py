"""
Plotting utilities for Voronoi-based cation-Cl cluster analysis.

This module provides plotting utilities inspired by cluster_analysis/utils/plots.py.
Functions are implemented locally for Phase 1 scope.
"""

from typing import Any, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

# -----------------------------
# Standard style config
# -----------------------------
PLOT_CONFIG = {
    'figure_size_large': (15, 10),
    'figure_size_medium': (12, 8),
    'figure_size_small': (10, 6),
    'figure_size_3d': (20, 15),
    'dpi': 100,
    'font_size_title': 16,
    'font_size_labels': 12,
    'font_size_legend': 10,
    'line_width': 2,
    'marker_size': 6,
    'alpha_main': 0.8,
    'alpha_background': 0.3,
    'grid_alpha': 0.3,
}

COLORS = {
    'species': {
        'Na': '#1f77b4',
        'Pu': '#d62728',
        'Cl': '#2ca02c',
        'Ce': '#9467bd',  # Added for future Ce support
        'unknown': '#7f7f7f'
    },
    'clusters': [
        '#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896'
    ],
    'coordination': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'background': '#f0f0f0',
    'grid': '#cccccc'
}


def setup_plot_style():
    """
    Configure matplotlib plotting style with standardized settings.
    
    Sets up default plotting parameters for consistent visualization
    across the analysis pipeline.
    """
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': PLOT_CONFIG['dpi'],
        'font.size': PLOT_CONFIG['font_size_labels'],
        'axes.titlesize': PLOT_CONFIG['font_size_title'],
        'axes.labelsize': PLOT_CONFIG['font_size_labels'],
        'xtick.labelsize': PLOT_CONFIG['font_size_labels'],
        'ytick.labelsize': PLOT_CONFIG['font_size_labels'],
        'legend.fontsize': PLOT_CONFIG['font_size_legend'],
        'lines.linewidth': PLOT_CONFIG['line_width'],
        'lines.markersize': PLOT_CONFIG['marker_size'],
        'grid.alpha': PLOT_CONFIG['grid_alpha'],
        'axes.grid': True,
        'axes.facecolor': COLORS['background'],
        'figure.facecolor': 'white',
    })


def get_standard_colors() -> Dict:
    """
    Get standard color scheme dictionary.
    
    Returns:
        Dictionary containing color schemes for species, clusters, coordination, etc.
    """
    return COLORS


# ============================================================================
# Phase 3: Weighted Coordination Plotting
# ============================================================================

def plot_coordination_histograms(
    histograms_dict: Dict[str, Any],
    cation: str,
    ax: Any = None,
    title: str = 'Coordination Histograms',
    show_statistics: bool = True,
) -> Any:
    """
    Plot side-by-side histograms for cation-Cl, cation-Na, cation-cation, and cation-Any coordination.
    
    This function implements Phase 3 plotting: Plot coordination histograms with statistics.
    
    Args:
        histograms_dict: Dictionary with keys 'pu_cl', 'pu_na', 'pu_pu', 'pu_any'
            Each value should be a dict with 'counts', 'bin_edges', and optionally 'values'
        cation: Cation species name (e.g., 'Pu', 'Ce') - required for labeling
        ax: Optional matplotlib axes object (if None, creates new figure)
        title: Plot title (default: 'Coordination Histograms')
        show_statistics: Whether to display mean ± std dev and median (default: True)
    
    Returns:
        matplotlib.axes.Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=PLOT_CONFIG['figure_size_medium'])
    
    # Extract histogram data
    pu_cl_data = histograms_dict.get('pu_cl', {'counts': np.array([]), 'bin_edges': np.array([])})
    pu_na_data = histograms_dict.get('pu_na', {'counts': np.array([]), 'bin_edges': np.array([])})
    pu_pu_data = histograms_dict.get('pu_pu', {'counts': np.array([]), 'bin_edges': np.array([])})
    pu_any_data = histograms_dict.get('pu_any', {'counts': np.array([]), 'bin_edges': np.array([])})
    
    # Get bin centers for plotting
    bins = histograms_dict.get('bins', pu_any_data.get('bin_edges', np.array([])))
    if len(bins) > 1:
        bin_centers = (bins[:-1] + bins[1:]) / 2
    else:
        bin_centers = np.array([])
    
    # Collect statistics for each type
    stats_text = []
    
    # Plot each histogram
    if len(bin_centers) > 0:
        width = 0.2  # Width of bars for grouped histogram
        
        # Helper function to compute statistics from values or counts
        def get_stats(data_dict, bin_centers):
            if 'values' in data_dict and len(data_dict['values']) > 0:
                values = data_dict['values']
                mean_val = np.mean(values)
                std_val = np.std(values)
                median_val = np.median(values)
            elif 'counts' in data_dict and len(data_dict['counts']) > 0:
                counts = data_dict['counts']
                if np.sum(counts) > 0:
                    # Compute weighted mean from histogram
                    mean_val = np.sum(counts * bin_centers) / np.sum(counts)
                    # Approximate std from histogram (less accurate)
                    variance = np.sum(counts * (bin_centers - mean_val)**2) / np.sum(counts)
                    std_val = np.sqrt(variance)
                    # Approximate median (find bin where cumulative sum crosses 0.5)
                    cumsum = np.cumsum(counts)
                    median_idx = np.searchsorted(cumsum, cumsum[-1] * 0.5)
                    median_val = bin_centers[median_idx] if median_idx < len(bin_centers) else bin_centers[-1]
                else:
                    return None
            else:
                return None
            return {'mean': mean_val, 'std': std_val, 'median': median_val}
        
        if len(pu_cl_data.get('counts', [])) > 0:
            ax.bar(bin_centers - 1.5*width, pu_cl_data['counts'], width, 
                   label=f'{cation}-Cl', color=COLORS['species']['Cl'], alpha=PLOT_CONFIG['alpha_main'])
            if show_statistics:
                stats = get_stats(pu_cl_data, bin_centers)
                if stats:
                    stats_text.append(f"{cation}-Cl: Mean={stats['mean']:.2f}±{stats['std']:.2f}, Median={stats['median']:.2f}")
        
        if len(pu_na_data.get('counts', [])) > 0:
            ax.bar(bin_centers - 0.5*width, pu_na_data['counts'], width,
                   label=f'{cation}-Na', color=COLORS['species']['Na'], alpha=PLOT_CONFIG['alpha_main'])
            if show_statistics:
                stats = get_stats(pu_na_data, bin_centers)
                if stats:
                    stats_text.append(f"{cation}-Na: Mean={stats['mean']:.2f}±{stats['std']:.2f}, Median={stats['median']:.2f}")
        
        if len(pu_pu_data.get('counts', [])) > 0:
            ax.bar(bin_centers + 0.5*width, pu_pu_data['counts'], width,
                   label=f'{cation}-{cation}', color=COLORS['species'].get(cation, COLORS['coordination'][0]), alpha=PLOT_CONFIG['alpha_main'])
            if show_statistics:
                stats = get_stats(pu_pu_data, bin_centers)
                if stats:
                    stats_text.append(f"{cation}-{cation}: Mean={stats['mean']:.2f}±{stats['std']:.2f}, Median={stats['median']:.2f}")
        
        if len(pu_any_data.get('counts', [])) > 0:
            ax.bar(bin_centers + 1.5*width, pu_any_data['counts'], width,
                   label=f'{cation}-Any', color='gray', alpha=PLOT_CONFIG['alpha_main'] * 0.7)
            if show_statistics:
                stats = get_stats(pu_any_data, bin_centers)
                if stats:
                    stats_text.append(f"{cation}-Any: Mean={stats['mean']:.2f}±{stats['std']:.2f}, Median={stats['median']:.2f}")
    
    # Add statistics text box
    if show_statistics and len(stats_text) > 0:
        stats_str = '\n'.join(stats_text)
        ax.text(0.95, 0.95, stats_str, transform=ax.transAxes,
                fontsize=PLOT_CONFIG['font_size_legend'], 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Get coordination type for title
    coord_type = histograms_dict.get('coordination_type', 'topological')
    coord_type_label = {
        'topological': 'Topological',
        'area': 'Area-weighted (CN_A)',
        'solid_angle': 'Solid-angle-weighted (CN_Ω)'
    }.get(coord_type, 'Topological')
    
    ax.set_xlabel('Coordination Number', fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_ylabel('Count', fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_title(f'{title} ({coord_type_label})', fontsize=PLOT_CONFIG['font_size_title'])
    ax.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
    
    return ax


def plot_weighted_coordination_comparison(
    cn_x: Dict[str, Any],
    cn_y: Dict[str, Any],
    species: np.ndarray,
    ax: Any = None,
    x_label: str = 'CN_A (Area-weighted)',
    y_label: str = 'CN_Ω (Solid-angle-weighted)',
    title: str = 'Weighted Coordination Comparison',
    show_statistics: bool = True,
    show_correlation: bool = True,
) -> Any:
    """
    Scatter plot comparing two coordination types for center species atoms.
    
    This function implements Phase 3 plotting: Plot weighted coordination comparison with statistics.
    
    Args:
        cn_x: Dictionary from coordination function with 'cn_values' and 'atom_indices' (x-axis)
        cn_y: Dictionary from coordination function with 'cn_values' and 'atom_indices' (y-axis)
        species: (N,) array of species names/identifiers
        ax: Optional matplotlib axes object (if None, creates new figure)
        x_label: Label for x-axis (default: 'CN_A (Area-weighted)')
        y_label: Label for y-axis (default: 'CN_Ω (Solid-angle-weighted)')
        title: Plot title (default: 'Weighted Coordination Comparison')
        show_statistics: Whether to display mean ± std dev and median (default: True)
        show_correlation: Whether to display correlation coefficient (default: True)
    
    Returns:
        matplotlib.axes.Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=PLOT_CONFIG['figure_size_medium'])
    
    # Extract CN values - need to align by atom indices
    cn_x_values = cn_x.get('cn_values', np.array([]))
    cn_x_indices = cn_x.get('atom_indices', np.array([], dtype=int))
    cn_y_values = cn_y.get('cn_values', np.array([]))
    cn_y_indices = cn_y.get('atom_indices', np.array([], dtype=int))
    
    # Find common atom indices
    if len(cn_x_indices) == 0 or len(cn_y_indices) == 0:
        ax.text(0.5, 0.5, 'No coordination data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(x_label, fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_ylabel(y_label, fontsize=PLOT_CONFIG['font_size_labels'])
        return ax
    
    # Create mapping from atom index to CN value
    cn_x_map = {idx: val for idx, val in zip(cn_x_indices, cn_x_values)}
    cn_y_map = {idx: val for idx, val in zip(cn_y_indices, cn_y_values)}
    
    # Find common indices
    common_indices = set(cn_x_map.keys()) & set(cn_y_map.keys())
    
    if len(common_indices) == 0:
        ax.text(0.5, 0.5, 'No common atoms found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(x_label, fontsize=PLOT_CONFIG['font_size_labels'])
        ax.set_ylabel(y_label, fontsize=PLOT_CONFIG['font_size_labels'])
        return ax
    
    # Extract aligned values
    aligned_cn_x = np.array([cn_x_map[idx] for idx in sorted(common_indices)])
    aligned_cn_y = np.array([cn_y_map[idx] for idx in sorted(common_indices)])
    
    # Create scatter plot
    # Use coordination color as fallback instead of hardcoded 'Pu'
    ax.scatter(aligned_cn_x, aligned_cn_y, 
               alpha=PLOT_CONFIG['alpha_main'], 
               s=PLOT_CONFIG['marker_size']**2,
               color=COLORS['coordination'][0],
               edgecolors='black', linewidth=0.5)
    
    # Add diagonal line for reference
    if len(aligned_cn_x) > 0:
        min_val = min(np.min(aligned_cn_x), np.min(aligned_cn_y))
        max_val = max(np.max(aligned_cn_x), np.max(aligned_cn_y))
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=PLOT_CONFIG['alpha_background'], 
               linewidth=1, label='y = x')
    
    # Calculate and display statistics
    stats_text = []
    if show_statistics and len(aligned_cn_x) > 0:
        mean_x = np.mean(aligned_cn_x)
        std_x = np.std(aligned_cn_x)
        median_x = np.median(aligned_cn_x)
        mean_y = np.mean(aligned_cn_y)
        std_y = np.std(aligned_cn_y)
        median_y = np.median(aligned_cn_y)
        
        stats_text.append(f"X: Mean={mean_x:.2f}±{std_x:.2f}, Median={median_x:.2f}")
        stats_text.append(f"Y: Mean={mean_y:.2f}±{std_y:.2f}, Median={median_y:.2f}")
    
    # Calculate correlation coefficient
    correlation = None
    if show_correlation and len(aligned_cn_x) > 1:
        if np.std(aligned_cn_x) > 0 and np.std(aligned_cn_y) > 0:
            correlation = np.corrcoef(aligned_cn_x, aligned_cn_y)[0, 1]
            stats_text.append(f"Correlation: r={correlation:.3f}")
    
    # Add statistics text box
    if len(stats_text) > 0:
        stats_str = '\n'.join(stats_text)
        ax.text(0.05, 0.95, stats_str, transform=ax.transAxes,
                fontsize=PLOT_CONFIG['font_size_legend'], 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel(x_label, fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_ylabel(y_label, fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_title(title, fontsize=PLOT_CONFIG['font_size_title'])
    ax.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
    
    return ax


# ============================================================================
# Phase 4: Graph Component Plotting
# ============================================================================

def plot_3d_cluster_component(
    graph: Any,
    cluster_ids: np.ndarray,
    cluster_id: int,
    positions: np.ndarray,
    species: np.ndarray,
    cation: str = 'Pu',
    anion: str = 'Cl',
    show_na_context: bool = True,
    show_edges: bool = True,
    scale_edge_width_by_area: bool = True,
    rotate_3d: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a specific cluster component in 3D showing both cation and anion atoms.
    
    This function implements Phase 4 plotting: Visualize a specific cluster by cluster_id.
    Unlike subplot-based functions, this plots a single cluster in one figure.
    
    Args:
        graph: networkx Graph from build_shared_anion_graph_from_voronoi
        cluster_ids: (N,) array of cluster IDs (-1 for unclustered)
        cluster_id: Specific cluster ID to plot (must be >= 0)
        positions: (N, 3) array of atom positions
        species: (N,) array of species names/identifiers
        cation: Cation species (default: 'Pu', can be 'Ce' or other)
        anion: Anion species (default: 'Cl')
        show_na_context: Whether to show Na atoms as background context (default: True)
        show_edges: Whether to draw edges between atoms (default: True)
        scale_edge_width_by_area: Whether to scale edge width by Voronoi face area (default: True).
            If False, all edges use uniform width of 1.5.
        title: Optional custom title (default: None, auto-generated)
        save_path: Optional path to save figure (default: None)
    
    Raises:
        ImportError: If networkx is not available
        ValueError: If cluster_id is invalid or cluster not found
    
    Examples:
        >>> plot_3d_cluster_component(
        ...     graph, cluster_ids, cluster_id=0,
        ...     positions, species, cation='Pu', anion='Cl'
        ... )
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for plot_3d_cluster_component")
    
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    if cluster_id < 0:
        raise ValueError(f"cluster_id must be >= 0, got {cluster_id}")
    
    # Find atoms in the specified cluster
    cluster_mask = (cluster_ids == cluster_id)
    cluster_atom_indices = np.where(cluster_mask)[0]
    
    if len(cluster_atom_indices) == 0:
        raise ValueError(f"Cluster {cluster_id} not found or empty")
    
    # Get subgraph for this cluster
    cluster_nodes = [int(idx) for idx in cluster_atom_indices if idx in graph.nodes()]
    if len(cluster_nodes) == 0:
        raise ValueError(f"No nodes from cluster {cluster_id} found in graph")
    
    cluster_subgraph = graph.subgraph(cluster_nodes)
    
    # Setup plot style
    setup_plot_style()
    
    # Enable 3D rotation
    if rotate_3d:
        plt.rcParams["axes3d.mouserotationstyle"] = "trackball"
    else:
        pass # Do nothing
    
    # Create figure with single 3D axes
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size_3d'])
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Na context if requested
    if show_na_context:
        na_mask = (species == 'Na')
        if np.any(na_mask):
            ax.scatter(
                positions[na_mask, 0],
                positions[na_mask, 1],
                positions[na_mask, 2],
                c=COLORS['species']['Na'],
                s=20,
                alpha=PLOT_CONFIG['alpha_background'],
                label='Na (context)',
            )
    
    # Separate atoms by species in cluster
    cation_positions = []
    anion_positions = []
    cation_indices = []
    anion_indices = []
    
    for node in cluster_subgraph.nodes():
        if node < len(positions) and node < len(species):
            pos = np.asarray(positions[node])
            if pos.shape[0] == 3:
                if species[node] == cation:
                    cation_positions.append(pos)
                    cation_indices.append(node)
                elif species[node] == anion:
                    anion_positions.append(pos)
                    anion_indices.append(node)
    
    # Plot cation atoms
    if len(cation_positions) > 0:
        cation_positions = np.vstack(cation_positions)
        cation_color = COLORS['species'].get(cation, COLORS['coordination'][0])
        ax.scatter(
            cation_positions[:, 0],
            cation_positions[:, 1],
            cation_positions[:, 2],
            c=cation_color,
            s=100,
            alpha=PLOT_CONFIG['alpha_main'],
            label=f'{cation} (n={len(cation_positions)})',
            edgecolors='darkred',
            linewidth=0.5,
        )
    
    # Plot anion atoms
    if len(anion_positions) > 0:
        anion_positions = np.vstack(anion_positions)
        ax.scatter(
            anion_positions[:, 0],
            anion_positions[:, 1],
            anion_positions[:, 2],
            c=COLORS['species'][anion],
            s=80,
            alpha=PLOT_CONFIG['alpha_main'],
            label=f'{anion} (n={len(anion_positions)})',
            edgecolors='darkgreen',
            linewidth=0.5,
        )
    
    # Plot edges if requested
    if show_edges and cluster_subgraph.number_of_edges() > 0:
        # Calculate edge widths based on area scaling if enabled
        if scale_edge_width_by_area:
            # Get edge areas for line width scaling
            areas = [edata.get("area", 1.0) for _, _, edata in cluster_subgraph.edges(data=True)]
            if len(areas) > 0:
                a_min = float(np.min(areas))
                a_max = float(np.max(areas))
            else:
                a_min = a_max = 1.0
            
            # Helper function for line width from area
            def lw_from_area(a: float, a0: float, a1: float) -> float:
                if a1 <= a0:
                    return 1.0
                t = (float(a) - a0) / (a1 - a0)
                return 0.6 + 2.4 * max(0.0, min(1.0, t))
        else:
            # Use uniform width for all edges
            uniform_width = 1.5
        
        # Draw edges
        for u, v, edata in cluster_subgraph.edges(data=True):
            if u < len(positions) and v < len(positions):
                p1 = np.asarray(positions[u])
                p2 = np.asarray(positions[v])
                if p1.shape[0] == 3 and p2.shape[0] == 3:
                    # Determine line width
                    if scale_edge_width_by_area:
                        area = edata.get("area", 1.0)
                        lw = lw_from_area(area, a_min, a_max)
                    else:
                        lw = uniform_width
                    
                    # Color edge based on species pair
                    u_species = species[u] if u < len(species) else 'unknown'
                    v_species = species[v] if v < len(species) else 'unknown'
                    
                    if (u_species == cation and v_species == anion) or \
                       (u_species == anion and v_species == cation):
                        edge_color = 'purple'  # Cation-anion edge
                    elif u_species == cation and v_species == cation:
                        edge_color = 'darkred'  # Cation-cation edge
                    elif u_species == anion and v_species == anion:
                        edge_color = 'darkgreen'  # Anion-anion edge
                    else:
                        edge_color = 'gray'
                    
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color=edge_color,
                        alpha=0.7,
                        linewidth=lw,
                    )
    
    # Set title
    if title is None:
        title = f"{cation}-{anion} Cluster {cluster_id}\n"
        title += f"({cation}: {len(cation_positions)}, {anion}: {len(anion_positions)}, "
        title += f"Edges: {cluster_subgraph.number_of_edges()})"
    
    ax.set_title(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    ax.set_xlabel("X (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_ylabel("Y (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_zlabel("Z (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
    ax.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()


def plot_unwrapped_3d_cluster_component(
    graph: Any,
    cluster_ids: np.ndarray,
    cluster_id: int,
    positions: np.ndarray,
    species: np.ndarray,
    cell_matrix: np.ndarray,
    cation: str = 'Pu',
    anion: str = 'Cl',
    show_na_context: bool = True,
    show_edges: bool = True,
    scale_edge_width_by_area: bool = True,
    rotate_3d: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a specific cluster component in 3D with unwrapped coordinates to prevent edges from spanning periodic boundaries.
    
    This function is similar to plot_3d_cluster_component() but uses coordinate unwrapping to ensure
    that edges between atoms in the cluster do not appear to "jump" across periodic boundaries.
    The unwrapping is performed using fractional coordinates and graph traversal to assign
    consistent integer lattice shifts to each node.
    
    Args:
        graph: networkx Graph from build_shared_anion_graph_from_voronoi
        cluster_ids: (N,) array of cluster IDs (-1 for unclustered)
        cluster_id: Specific cluster ID to plot (must be >= 0)
        positions: (N, 3) array of wrapped atom positions
        species: (N,) array of species names/identifiers
        cell_matrix: (3, 4) or (3, 3) cell matrix from OVITO (required)
            - If (3, 4): first 3 columns are lattice vectors, 4th is origin offset
            - If (3, 3): columns are lattice vectors
        cation: Cation species (default: 'Pu', can be 'Ce' or other)
        anion: Anion species (default: 'Cl')
        show_na_context: Whether to show Na atoms as background context (default: True)
            Note: Na atoms are shown with wrapped coordinates
        show_edges: Whether to draw edges between atoms (default: True)
        scale_edge_width_by_area: Whether to scale edge width by Voronoi face area (default: True).
            If False, all edges use uniform width of 1.5.
        title: Optional custom title (default: None, auto-generated)
        save_path: Optional path to save figure (default: None)
    
    Raises:
        ImportError: If networkx is not available
        ValueError: If cluster_id is invalid or cluster not found
    
    Examples:
        >>> plot_unwrapped_3d_cluster_component(
        ...     graph, cluster_ids, cluster_id=0,
        ...     positions, species, cell_matrix, cation='Pu', anion='Cl'
        ... )
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for plot_unwrapped_3d_cluster_component")
    
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from utils import unwrap_component
    
    if cluster_id < 0:
        raise ValueError(f"cluster_id must be >= 0, got {cluster_id}")
    
    # Find atoms in the specified cluster
    cluster_mask = (cluster_ids == cluster_id)
    cluster_atom_indices = np.where(cluster_mask)[0]
    
    if len(cluster_atom_indices) == 0:
        raise ValueError(f"Cluster {cluster_id} not found or empty")
    
    # Get subgraph for this cluster
    cluster_nodes = [int(idx) for idx in cluster_atom_indices if idx in graph.nodes()]
    if len(cluster_nodes) == 0:
        raise ValueError(f"No nodes from cluster {cluster_id} found in graph")
    
    cluster_subgraph = graph.subgraph(cluster_nodes)
    
    # Unwrap coordinates for the cluster component
    R_unwrapped, index_of = unwrap_component(
        positions=positions,
        cell_matrix=cell_matrix,
        graph=graph,
        component_nodes=cluster_nodes,
        species=species,
        cation=cation,
    )
    
    # Setup plot style
    setup_plot_style()
    
    # Enable 3D rotation
    if rotate_3d:
        plt.rcParams["axes3d.mouserotationstyle"] = "trackball"
    else:
        pass # Do nothing
    
    # Create figure with single 3D axes
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size_3d'])
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Na context if requested (using wrapped coordinates)
    if show_na_context:
        na_mask = (species == 'Na')
        if np.any(na_mask):
            ax.scatter(
                positions[na_mask, 0],
                positions[na_mask, 1],
                positions[na_mask, 2],
                c=COLORS['species']['Na'],
                s=20,
                alpha=PLOT_CONFIG['alpha_background'],
                label='Na (context)',
            )
    
    # Separate atoms by species in cluster and use unwrapped positions
    cation_positions = []
    anion_positions = []
    cation_indices = []
    anion_indices = []
    
    for node in cluster_subgraph.nodes():
        if node in index_of:
            idx = index_of[node]
            pos = R_unwrapped[idx]
            if pos.shape[0] == 3:
                if species[node] == cation:
                    cation_positions.append(pos)
                    cation_indices.append(node)
                elif species[node] == anion:
                    anion_positions.append(pos)
                    anion_indices.append(node)
    
    # Plot cation atoms with unwrapped positions
    if len(cation_positions) > 0:
        cation_positions = np.vstack(cation_positions)
        cation_color = COLORS['species'].get(cation, COLORS['coordination'][0])
        ax.scatter(
            cation_positions[:, 0],
            cation_positions[:, 1],
            cation_positions[:, 2],
            c=cation_color,
            s=100,
            alpha=PLOT_CONFIG['alpha_main'],
            label=f'{cation} (n={len(cation_positions)})',
            edgecolors='darkred',
            linewidth=0.5,
        )
    
    # Plot anion atoms with unwrapped positions
    if len(anion_positions) > 0:
        anion_positions = np.vstack(anion_positions)
        ax.scatter(
            anion_positions[:, 0],
            anion_positions[:, 1],
            anion_positions[:, 2],
            c=COLORS['species'][anion],
            s=80,
            alpha=PLOT_CONFIG['alpha_main'],
            label=f'{anion} (n={len(anion_positions)})',
            edgecolors='darkgreen',
            linewidth=0.5,
        )
    
    # Plot edges if requested (using unwrapped positions)
    if show_edges and cluster_subgraph.number_of_edges() > 0:
        # Calculate edge widths based on area scaling if enabled
        if scale_edge_width_by_area:
            # Get edge areas for line width scaling
            areas = [edata.get("area", 1.0) for _, _, edata in cluster_subgraph.edges(data=True)]
            if len(areas) > 0:
                a_min = float(np.min(areas))
                a_max = float(np.max(areas))
            else:
                a_min = a_max = 1.0
            
            # Helper function for line width from area
            def lw_from_area(a: float, a0: float, a1: float) -> float:
                if a1 <= a0:
                    return 1.0
                t = (float(a) - a0) / (a1 - a0)
                return 0.6 + 2.4 * max(0.0, min(1.0, t))
        else:
            # Use uniform width for all edges
            uniform_width = 1.5
        
        # Draw edges using unwrapped positions
        for u, v, edata in cluster_subgraph.edges(data=True):
            if u in index_of and v in index_of:
                idx_u = index_of[u]
                idx_v = index_of[v]
                p1 = R_unwrapped[idx_u]
                p2 = R_unwrapped[idx_v]
                if p1.shape[0] == 3 and p2.shape[0] == 3:
                    # Determine line width
                    if scale_edge_width_by_area:
                        area = edata.get("area", 1.0)
                        lw = lw_from_area(area, a_min, a_max)
                    else:
                        lw = uniform_width
                    
                    # Color edge based on species pair
                    u_species = species[u] if u < len(species) else 'unknown'
                    v_species = species[v] if v < len(species) else 'unknown'
                    
                    if (u_species == cation and v_species == anion) or \
                       (u_species == anion and v_species == cation):
                        edge_color = 'purple'  # Cation-anion edge
                    elif u_species == cation and v_species == cation:
                        edge_color = 'darkred'  # Cation-cation edge
                    elif u_species == anion and v_species == anion:
                        edge_color = 'darkgreen'  # Anion-anion edge
                    else:
                        edge_color = 'gray'
                    
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color=edge_color,
                        alpha=0.7,
                        linewidth=lw,
                    )
    
    # Set title
    if title is None:
        title = f"{cation}-{anion} Cluster {cluster_id} (Unwrapped)\n"
        title += f"({cation}: {len(cation_positions)}, {anion}: {len(anion_positions)}, "
        title += f"Edges: {cluster_subgraph.number_of_edges()})"
    
    ax.set_title(title, fontsize=PLOT_CONFIG['font_size_title'], fontweight='bold')
    ax.set_xlabel("X (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_ylabel("Y (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
    ax.set_zlabel("Z (Å)", fontsize=PLOT_CONFIG['font_size_labels'])
    ax.legend(fontsize=PLOT_CONFIG['font_size_legend'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    
    plt.show()



