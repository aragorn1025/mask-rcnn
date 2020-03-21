import argparse
import cv2
import os

import utilities.engine
import utilities.models.mask_rcnn
import utilities.tools.file
import utilities.tools.general
import utilities.tools.instance_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Instance detection using Mask R-CNN.')
    parser.add_argument('-i', '--input', type = str,
        help = 'The path of file.')
    parser.add_argument('-o', '--output', type = str, default = None,
        help = 'The path to save.')
    parser.add_argument('-c', '--classes', type = str, default = 'data/classes/classes.names',
        help = 'The names of the classes.')
    parser.add_argument('-w', '--weights', type = str, default = 'weights/weights.pth',
        help = 'The weights to loaded.')
    parser.add_argument('--threshold', type = float, default = 0.8,
        help = 'The threshold for the mask.')
    parser.add_argument('--to_print_message', dest = 'to_print_message', action = 'store_true',
        help = 'To print message at the terminal.')
    parser.set_defaults(to_print_message = False)
    parser.add_argument('--device', type = str, default = 'cuda',
        help = 'Choose the device to use.')
    args = vars(parser.parse_args())
    for key in ['input', 'classes', 'weights']:
        utilities.tools.file.check_file(key, args[key])
    if args['output'] == None:
        args['output'] = './outputs/%s.png' % os.path.splitext(os.path.basename(args['input']))[0]
    utilities.tools.file.check_output(args['output'])
    
    class_names = utilities.tools.general.get_classes(args['classes'])
    engine = utilities.engine.Engine(
        model = utilities.models.mask_rcnn.MaskRCNN(number_classes = len(class_names)),
        device = args['device']
    )
    engine.load(args['weights'])
    image = cv2.imread(args['input'])
    masks, boxes, labels = utilities.tools.instance_segmentation.get_predictions(engine, class_names, image, threshold = args['threshold'])
    if args['to_print_message']:
        print(utilities.tools.general.get_predictions_number_message(len(labels)))
    cv2.imwrite(args['output'], utilities.tools.instance_segmentation.get_masked_image(image, (masks, boxes, labels)))
