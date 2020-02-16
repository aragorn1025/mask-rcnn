import argparse
import cv2
import os

import utilities.engine
import utilities.models.mask_rcnn
import utilities.tools.general
import utilities.tools.instance_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Instance detection using Mask R-CNN.')
    parser.add_argument('--weights', type = str, default = './weights/weights.pth',
        help = 'The weights to loaded')
    parser.add_argument('--classes', type = str, default = './data/classes.names',
        help = 'The names of the classes')
    parser.add_argument('--input', type = str,
        help = 'The path of file')
    parser.add_argument('--output', type = str, default = './outputs/output.png',
        help = 'The path to save')
    parser.add_argument('--threshold', type = float, default = 0.8,
        help = 'Threshold for the mask')
    parser.add_argument('--to_print_message', dest = 'to_print_message', action = 'store_true',
        help = 'to print message at the terminal')
    parser.set_defaults(to_print_message = False)
    parser.add_argument('--to_use_gpu', dest = 'to_use_gpu', action = 'store_true',
        help = 'to use GPU if available')
    parser.add_argument('--to_use_cpu', dest = 'to_use_gpu', action = 'store_false',
        help = 'to use CPU rather than GPU')
    parser.set_defaults(to_use_gpu = True)
    args = vars(parser.parse_args())
    for k in ['weights', 'classes', 'input']:
        if args[k] == None:
            raise ValueError('Thw %s file path should be set.' % k)
        if not os.path.isfile(args[k]):
            raise IOError('The %s file is not found.' % k)

    engine = utilities.engine.Engine(
        model = utilities.models.mask_rcnn.MaskRCNN(number_classes = 10 + 1),
        device = args['to_use_gpu']
    )
    engine.load(args['weights'])
    class_names = utilities.tools.general.get_classes(args['classes'])
    image = cv2.imread(args['input'])
    masks, boxes, labels = utilities.tools.instance_segmentation.get_predictions(engine, class_names, image, threshold = args['threshold'])
    if args['to_print_message']:
        utilities.tools.general.print_predictions_number(len(labels))
    cv2.imwrite(args['output'], utilities.tools.instance_segmentation.get_masked_image(image, (masks, boxes, labels)))