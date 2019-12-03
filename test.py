import argparse
import numpy as np
import os

import utilities
from utilities.models.MaskRCNN import MaskRCNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Instance detection using Mask R-CNN .')
    parser.add_argument('--weights', type=str, default="weights/weights.pth",
        help='The weights to loaded')
    parser.add_argument('--inputs', type=str,
        help='The inputs')
    parser.add_argument('--outputs', type=str,
        help='The outputs')
    args = vars(parser.parse_args())

    if args['weights'] == None:
        raise ValueError('Thw weights file path should be set.')
    if not os.path.isfile(args['weights']):
        raise IOError('The weights file is not found.')
    if args['inputs'] == None:
        raise ValueError('The inputs file path should be set.')
    if not os.path.isfile(args['inputs']):
        raise IOError('The inputs file is not found.')
    if args['outputs'] == None:
        raise ValueError('The outputs file path should be set.')

    engine = utilities.Engine(model = MaskRCNN())
    engine.load(args['weights'])

	# load inputs
	#inputs = 
	raise NotImplementedError('The codes for loading inputs is not implement yet.')

	# save output
    #outputs = engine.get_outputs(inputs)
	raise NotImplementedError('The codes for saveing outputs is not implement yet.')
