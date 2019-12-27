import cv2
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

def get_masked_image(image, predictions, rectangle_thickness = 2, text_size = 1, text_thickness = 2):
    masks, boxes, labels = predictions
    result = np.copy(image)
    for i in range(0, len(labels)):
        colored_mask = get_random_colored_mask(masks[i])
        result = cv2.addWeighted(result, 1, colored_mask, 0.5, 0)
        cv2.rectangle(result, boxes[i][0], boxes[i][1], color = (0, 255, 0), thickness = rectangle_thickness)
        cv2.putText(result, str(labels[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness = text_thickness)
    return result

def get_random_colored_mask(mask, colors = None):
    return get_colored_mask(mask, random.choice(colors if colors else _colors))