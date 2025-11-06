"""
Plotting utilities for Voronoi-based Pu-Cl cluster analysis.

This module provides plotting utilities inspired by cluster_analysis/utils/plots.py.
Functions are implemented locally for Phase 1 scope.
"""

from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt

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



