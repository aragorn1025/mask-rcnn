import cv2
import numpy as np
import random
import base64
import json

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

def get_masked_image(image, predictions, rectangle_thickness = 1, text_size = 1, text_thickness = 2):
    masks, boxes, labels = predictions
    result = np.copy(image)
    for i in range(0, len(labels)):
        colored_mask = get_random_colored_mask(masks[i])
        result = cv2.addWeighted(result, 1, colored_mask, 0.5, 0)
        cv2.rectangle(result, boxes[i][0], boxes[i][1], color = (0, 255, 0), thickness = rectangle_thickness)
        cv2.putText(result, str(labels[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness = text_thickness)
    return result

def get_predictions(engine, class_names, inputs, is_tensor = False, threshold = 0.8):
    predictions = engine.get_outputs(inputs, is_tensor)[0]
    scores = list(predictions['scores'].detach().cpu().numpy())

    predictions_pass_threshold = [scores.index(x) for x in scores if x > threshold]
    if len(predictions_pass_threshold) == 0:
        return None, None, []
    predictions_pass_threshold_last_index = [scores.index(x) for x in scores if x > threshold][-1]

    masks = (predictions['masks'] > 0.5).squeeze().detach().cpu().numpy()
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions['boxes'].detach().cpu().numpy())]
    labels = [class_names[i] for i in list(predictions['labels'].cpu().numpy())]

    if predictions_pass_threshold_last_index > 0:
        masks = masks[:predictions_pass_threshold_last_index + 1]
    elif len(masks.shape) == 2:
        masks = np.asarray([masks])
    boxes = boxes[:predictions_pass_threshold_last_index + 1]
    labels = labels[:predictions_pass_threshold_last_index + 1]    
    return masks, boxes, labels

def get_masks_polygon(masks):
    masks_list, masks_list_0, masks_list_1, masks_list_2 = [], [], [], []
    for i in range(len(masks[0])):
        for j in range(len(masks[0][0])):
            if masks[0][i,j] == True:
                masks_list_0.append(float(i)), masks_list_1.append(j)            
    masks_list_0 = np.unique(masks_list_0)
    masks_list_2.append(float(masks_list_1[0]))
    for k in range(len(masks_list_1)-1):
        if masks_list_1[k] > masks_list_1[k+1]:
            masks_list_2.append(float(masks_list_1[k])), masks_list_2.append(float(masks_list_1[k+1]))
    masks_list_2.append(float(masks_list_1[-1]))
    masks_list_0 = [x for pair in zip(masks_list_0, masks_list_0) for x in pair]
    masks_list = list(zip(masks_list_2, masks_list_0))
    for i in range(len(masks_list)):
        masks_list[i] = list(masks_list[i])
    return masks_list

def get_mask_json(image_path, masks_list, width, height):
    image_path = str(image_path).lstrip("{'image_path'): '").rstrip("'}")
    print(image_path)
    with open(image_path, 'rb') as g:
        imageData = g.read()
        imageData = base64.b64encode(imageData).decode('utf-8')

    abc = {"version": "3.16.7",
           "flag": {},
           "shapes": [
               {
                   "label": "1",
                   "line_color": None,
                   "fill_color": None,
                   "points": masks_list,
                   "shape_type": "polygon",
                   "flags": {}    
               }
           ],
           "lineColor": [0,255,0,128],
           "fillColor": [255,0,0,128],
           "imagePath": str(image_path),
           "imageData": str(imageData),
           "imageHeight": width,
           "imageWidth": height
          }
    with open(str(image_path).rstrip('.bmp') + '.json', 'w') as f:
        json.dump(abc, f)

def get_random_colored_mask(mask, colors = None):
    return get_colored_mask(mask, random.choice(colors if colors else _colors))