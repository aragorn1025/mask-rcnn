import argparse
import cv2
import os

import utilities.engine
import utilities.models.mask_rcnn
import utilities.tools.file
import utilities.tools.general
import utilities.tools.instance_segmentation

def main(inputs, outputs, clazz, weights, threshold, device, to_print_message):
    utilities.tools.file.check_file('inputs', inputs)
    if outputs == None:
        outputs = './outputs/%s.png' % os.path.splitext(os.path.basename(inputs))[0]
    utilities.tools.file.check_outputs(outputs)
    utilities.tools.file.check_file('classes', clazz)
    utilities.tools.file.check_file('weights', weights)
    
    class_names = utilities.tools.general.get_classes(clazz)
    engine = utilities.engine.Engine(
        model = utilities.models.mask_rcnn.MaskRCNN(number_classes = len(class_names)),
        device = device
    )
    engine.load(weights)
    image = cv2.imread(inputs)
    masks, boxes, labels = utilities.tools.instance_segmentation.get_predictions(engine, class_names, image, threshold = threshold)
    if to_print_message:
        print(utilities.tools.general.get_predictions_number_message(len(labels)))
    cv2.imwrite(outputs, utilities.tools.instance_segmentation.get_masked_image(image, (masks, boxes, labels)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Instance detection using Mask R-CNN.')
    parser.add_argument('-i', '--inputs', type = str,
        help = 'The root of the inputs.')
    parser.add_argument('-o', '--outputs', type = str, default = None,
        help = 'The root of the outputs.')
    parser.add_argument('-c', '--class', type = str, default = 'data/classes/data.class',
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
    
    main(args['inputs'], args['outputs'], args['class'], args['weights'], args['threshold'], args['device'], args['to_print_message'])
