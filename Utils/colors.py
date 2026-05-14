"""
Color-blind friendly color palette for consistent plotting across the project.

Usage:
    from Utils.colors import COLORBLIND_COLORS
    
    plt.plot(x, y, color=COLORBLIND_COLORS['orange'])
    
Or import hex codes:
    from Utils.colors import COLORBLIND_COLORS_HEX
    
    plt.plot(x, y, color=COLORBLIND_COLORS_HEX['orange'])
"""

# Colors as RGB tuples (0-255 range)
COLORBLIND_COLORS_RGB = {
    'black': (0, 0, 0),
    'orange': (230, 150, 0),
    'light_blue': (86, 180, 233),
    'green': (0, 158, 115),
    'yellow': (240, 228, 66),
    'blue': (0, 114, 178),
    'red': (213, 84, 0),
    'pink': (204, 121, 167),
}

# Colors as matplotlib-friendly normalized RGB (0-1 range)
COLORBLIND_COLORS = {
    'black': (0/255, 0/255, 0/255),
    'orange': (230/255, 150/255, 0/255),
    'light_blue': (86/255, 180/255, 233/255),
    'green': (0/255, 158/255, 115/255),
    'yellow': (240/255, 228/255, 66/255),
    'blue': (0/255, 114/255, 178/255),
    'red': (213/255, 84/255, 0/255),
    'pink': (204/255, 121/255, 167/255),
}

# Hex codes for convenience
COLORBLIND_COLORS_HEX = {
    'black': '#000000',
    'orange': '#E69600',
    'light_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'red': '#D55400',
    'pink': '#CC79A7',
}

# Common combinations for plots with two variables
NUCLEOSOME_CENTROMERE = {
    'nucleosome': COLORBLIND_COLORS['red'],
    'centromere': COLORBLIND_COLORS['green'],
}

NUCLEOSOME_CENTROMERE_HEX = {
    'nucleosome': COLORBLIND_COLORS_HEX['red'],
    'centromere': COLORBLIND_COLORS_HEX['green'],
}

# Palette as a list for multiple categories
COLORBLIND_PALETTE = [
    COLORBLIND_COLORS['blue'],
    COLORBLIND_COLORS['orange'],
    COLORBLIND_COLORS['green'],
    COLORBLIND_COLORS['red'],
    COLORBLIND_COLORS['light_blue'],
    COLORBLIND_COLORS['pink'],
    COLORBLIND_COLORS['yellow'],
    COLORBLIND_COLORS['black'],
]

COLORBLIND_PALETTE_HEX = [
    COLORBLIND_COLORS_HEX['blue'],
    COLORBLIND_COLORS_HEX['orange'],
    COLORBLIND_COLORS_HEX['green'],
    COLORBLIND_COLORS_HEX['red'],
    COLORBLIND_COLORS_HEX['light_blue'],
    COLORBLIND_COLORS_HEX['pink'],
    COLORBLIND_COLORS_HEX['yellow'],
    COLORBLIND_COLORS_HEX['black'],
]
