import argparse
import cv2
import numpy as np
import os
import torch

import utilities.engine
import utilities.models.mask_rcnn
import utilities.tools.general
import utilities.tools.instance_segmentation

def get_predictions(engine, category_names, image, threshold):
    predictions = engine.get_outputs(image)[0]
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
    classes = utilities.tools.general.get_classes(args['classes'])
    image = cv2.imread(args['input'])
    masks, boxes, labels = get_predictions(engine, classes, image, args['threshold'])
    if args['to_print_message']:
        utilities.tools.general.print_predictions_number(len(labels))
    cv2.imwrite(args['output'], utilities.tools.instance_segmentation.get_masked_image(image, (masks, boxes, labels)))