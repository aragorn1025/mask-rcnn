import numpy as np
import random

_colors = [
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
    [80, 70, 180],
    [250, 80, 190],
    [245, 145, 50],
    [70, 150, 250],
    [50, 190, 190]
]

def get_colored_mask(mask, color):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask

def get_random_colored_mask(mask, colors = None):
    return get_colored_mask(mask, random.choice(colors if colors else _colors))