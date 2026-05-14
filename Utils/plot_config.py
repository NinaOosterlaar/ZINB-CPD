"""
Standardized plotting configuration for consistent figures across the project.

Usage:
    from Utils.plot_config import setup_plot_style
    
    # Call at the beginning of your script
    setup_plot_style()
    
Or import colors directly:
    from Utils.plot_config import COLORS
    plt.plot(x, y, color=COLORS['blue'])
"""

import matplotlib.pyplot as plt
from Utils.colors import COLORBLIND_COLORS_HEX, COLORBLIND_COLORS, NUCLEOSOME_CENTROMERE_HEX

# Export colors for easy access
COLORS = COLORBLIND_COLORS_HEX
COLORS_NORMALIZED = COLORBLIND_COLORS
NUCLEOSOME_CENTROMERE = NUCLEOSOME_CENTROMERE_HEX

def setup_plot_style():
    """
    Configure matplotlib with standardized font sizes and style settings.
    Call this at the beginning of your plotting scripts.
    """
    plt.rcParams.update({
        'font.size': 14,          # Base font size
        'axes.labelsize': 14,     # X and Y labels
        'axes.titlesize': 20,     # Subplot titles
        'xtick.labelsize': 10,    # X-axis tick labels
        'ytick.labelsize': 14,    # Y-axis tick labels
        'legend.fontsize': 14,    # Legend
        'figure.titlesize': 24    # Main title (suptitle)
    })
