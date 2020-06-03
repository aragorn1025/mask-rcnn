import cv2
import numpy as np
import torchvision
import PIL
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

def get_crowd(file_name):
    with open(file_name, 'r') as file:
        rows = [r.strip() for r in file.read().split('\n') if len(r.strip()) > 0]
    crowd = [False] + [r == 'True' for r in rows]
    return crowd

def _get_crowded_box(box_0, box_1):
    xmin = min(box_0[0][0], box_1[0][0])
    ymin = min(box_0[0][1], box_1[0][1])
    xmax = max(box_0[1][0], box_1[1][0])
    ymax = max(box_0[1][1], box_1[1][1])
    return (xmin, ymin), (xmax, ymax)

def get_crowded_predictions(predictions, clazz, crowd):
    masks, boxes, labels = predictions
    if len(labels) <= 1:
        return masks, boxes, labels
    crowdable_clazz = []
    for cl, cr in zip(clazz, crowd):
        if cr:
            crowdable_clazz.append(cl)
    crowding_masks = {}
    crowding_boxes = {}
    crowded_masks = []
    crowded_boxes = []
    crowded_labels = []
    for mask, box, label in zip(masks, boxes, labels):
        if label not in crowdable_clazz:
            crowded_masks.append(mask)
            crowded_boxes.append(box)
            crowded_labels.append(label)
        else:
            if label not in crowding_masks.keys():
                crowding_masks[label] = mask
                crowding_boxes[label] = box
            crowding_masks[label] = np.bitwise_or(crowding_masks[label], mask)
            crowding_boxes[label] = _get_crowded_box(crowding_boxes[label], box)
    for label in crowding_masks.keys():
        crowded_masks.append(crowding_masks[label])
        crowded_boxes.append(crowding_boxes[label])
        crowded_labels.append(label)
    return crowded_masks, crowded_boxes, crowded_labels

def get_masked_image(image, predictions, rectangle_thickness = 1, text_size = 1, text_thickness = 2, mask_weight = 0.5, mask_colors = None):
    masks, boxes, labels = predictions
    result = np.copy(image)
    for i in range(0, len(labels)):
        colored_mask = get_random_colored_mask(masks[i], colors = mask_colors)
        result = cv2.addWeighted(result, 1, colored_mask, mask_weight, 0)
        if rectangle_thickness <= 0:
            continue
        cv2.rectangle(result, boxes[i][0], boxes[i][1], color = (0, 255, 0), thickness = rectangle_thickness)
        if text_size <= 0 or text_thickness <= 0:
            continue
        cv2.putText(result, str(labels[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness = text_thickness)
    return result

def get_merged_predictions(predictions):
    masks, boxes, labels = predictions
    print(masks)
    print(boxes)
    print(labels)

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

def get_random_colored_mask(mask, colors = None):
    return get_colored_mask(mask, random.choice(colors if colors else _colors))

def get_transforms(resized_size):
    transforms = {}
    if resized_size == None or resized_size[0] <= 0 or resized_size[1] <= 0:
        transforms['images'] = None
        transforms['masks'] = None
    else:
        transforms['images'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resized_size),
            torchvision.transforms.ToTensor(),
        ])
        transforms['masks'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resized_size, PIL.Image.NEAREST)
        ])
    return transforms

####################################################################################################

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
