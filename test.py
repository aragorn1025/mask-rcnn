import argparse
import cv2
import numpy as np
import os
import torch
import torchvision

import utilities.engine
import utilities.models.mask_rcnn
import utilities.tools.general
import utilities.tools.instance_segmentation

def get_predictions(image, engine, threshold, category_names):
    image = torchvision.transforms.ToTensor()(image)
    predictions = engine.get_outputs([image])[0]
    scores = list(predictions['scores'].detach().cpu().numpy())

    predictions_pass_threshold = [scores.index(x) for x in scores if x > threshold]
    if len(predictions_pass_threshold) == 0:
        return None, None, []
    predictions_pass_threshold_last_index = [scores.index(x) for x in scores if x > threshold][-1]

    masks = (predictions['masks'] > 0.5).squeeze().detach().cpu().numpy()
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions['boxes'].detach().cpu().numpy())]
    labels = [category_names[i] for i in list(predictions['labels'].cpu().numpy())]

    if predictions_pass_threshold_last_index > 0:
        masks = masks[:predictions_pass_threshold_last_index + 1]
    elif len(masks.shape) == 2:
        masks = np.asarray([masks])
    boxes = boxes[:predictions_pass_threshold_last_index + 1]
    labels = labels[:predictions_pass_threshold_last_index + 1]    
    return masks, boxes, labels

def print_message(n):
    if n < 1:
        print("There is no prediction.")
    elif n == 1:
        print("There is 1 prediction.")
    else:
        print("There are %d predictions." % n)

def get_masked_image(image, predictions, rectangle_thickness = 2, text_size = 1, text_thickness = 2):
    masks, boxes, labels = predictions
    result = np.copy(image)
    for i in range(0, len(labels)):
        colored_mask = utilities.tools.instance_segmentation.get_random_colored_mask(masks[i])
        result = cv2.addWeighted(result, 1, colored_mask, 0.5, 0)
        cv2.rectangle(result, boxes[i][0], boxes[i][1], color = (0, 255, 0), thickness = rectangle_thickness)
        cv2.putText(result, str(labels[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness = text_thickness)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Instance detection using Mask R-CNN.')
    parser.add_argument('--weights', type=str, default = './weights/weights.pth',
        help='The weights to loaded')
    parser.add_argument('--classes', type=str, default = './data/classes.names',
        help='The names of the classes')
    parser.add_argument('--input', type=str,
        help='The path of file')
    parser.add_argument('--output', type=str, default = './outputs/output.png',
        help='The path to save')
    parser.add_argument('--threshold', type=float, default = 0.8,
        help='Threshold for the mask')
    parser.add_argument('--to_print_message', dest='to_print_message', action='store_true',
        help='to print message at the terminal')
    parser.set_defaults(to_print_message=False)
    parser.add_argument('--to_use_gpu', dest='to_use_gpu', action='store_true',
        help='to use GPU if available')
    parser.add_argument('--to_use_cpu', dest='to_use_gpu', action='store_false',
        help='use CPU rather than GPU')
    parser.set_defaults(to_use_gpu=True)
    args = vars(parser.parse_args())
    for k in ['weights', 'classes', 'input']:
        if args[k] == None:
            raise ValueError('Thw %s file path should be set.' % k)
        if not os.path.isfile(args[k]):
            raise IOError('The %s file is not found.' % k)

    engine = utilities.engine.Engine(
        model = utilities.models.mask_rcnn.MaskRCNN(number_classes = 10 + 1),
        device = None if args['to_use_gpu'] else torch.device('cpu')
    )
    engine.load(args['weights'])
    image = cv2.imread(args['input'])
    classes = utilities.tools.general.get_classes(args['classes'])

    masks, boxes, labels = get_predictions(image, engine, args['threshold'], classes)
    n = len(labels)
    if args['to_print_message']:
        print_message(n)
    result = get_masked_image(image, (masks, boxes, labels))
    cv2.imwrite(args['output'], result)