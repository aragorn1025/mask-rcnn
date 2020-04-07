import argparse
import math
import os
import torch

import thirdparty.torchvision.detection.utils as tutils
import thirdparty.torchvision.detection.engine as tengine
import utilities.datasets as udatasets
import utilities.engine as uengine
import utilities.models as umodels
import utilities.tools as utools

def _get_dataset(dataset_type, root_images, root_masks, image_extension = '', class_names = [], is_crowd = [], transforms = None):
    if dataset_type == 'cityscapes':
        return udatasets.cityscapes_dataset.CityscapesDataset(root_images, root_masks, transforms['images'], transforms['masks'])
    if dataset_type == 'label_me':
        return udatasets.label_me_dataset.LabelMeDataset(root_images, root_masks, image_extension, class_names, is_crowd, transforms['images'], transforms['masks'])
    raise NotImplementedError('Unknown dataset type.')

def _get_checkouts_format(weights, checkouts, bits = -1):
    w = os.path.splitext(os.path.basename(weights))
    if bits <= 0:
        return os.path.join(checkouts, '%s_%%d%s' % (w[0], w[1]))
    else:
        return os.path.join(checkouts, '%s_%%0%dd%s' % (w[0], bits, w[1]))

def main(dataset_type, root_dataset, clazz, crowd, weights, checkouts, epoch, device, resized_size, batch_size, learning_rate, print_frequency, to_skip_evaluation):
    for key in ['train_image', 'train_mask', 'test_image', 'test_mask']:
        utools.file.check_directory(key, root_dataset[key])
    if weights == None:
        weights = 'weights/weights.pth'
    utools.file.check_output(weights)
    if checkouts != None:
        utools.file.check_output(os.path.join(checkouts, 'weights.pth'))
        checkout_format = _get_checkouts_format(weights, checkouts, math.ceil(math.log10(epoch)))
    
    dataset = {}
    data_loader = {}
    transforms = utools.instance_segmentation.get_transforms(resized_size)
    class_names = utools.general.get_classes(clazz)
    if crowd == None:
        is_crowd = [False] * len(class_names)
    else:
        is_crowd = utools.instance_segmentation.get_crowd(crowd)
    for key in ['train', 'test']:
        dataset[key] = _get_dataset(
            dataset_type = dataset_type,
            root_images = root_dataset['%s_image' % key],
            root_masks = root_dataset['%s_mask' % key],
            image_extension = root_dataset['image_extension'],
            class_names = class_names,
            is_crowd = is_crowd,
            transforms = transforms
        )
        print('There are %d %sing images.' % (len(dataset[key]), key))
        data_loader[key] = torch.utils.data.DataLoader(
            dataset[key],
            batch_size = batch_size,
            shuffle = (key == 'train'),
            num_workers = 1,
            collate_fn = tutils.collate_fn
        )
    
    model = umodels.mask_rcnn.MaskRCNN(number_classes = len(class_names))
    parameters = [p for p in model.parameters() if p.requires_grad]
    engine = uengine.Engine(
        model = model,
        criterion = None,
        optimizer = torch.optim.SGD(parameters, lr = learning_rate, momentum = 0.9, weight_decay = 0.0005),
        device = device
    )
    if os.path.isfile(weights):
        engine.load(weights)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(engine._optimizer, step_size = 3, gamma = 0.1)
    for i in range(0, epoch):
        tengine.train_one_epoch(engine._model, engine._optimizer, data_loader['train'], engine._device, i, print_freq = print_frequency)
        lr_scheduler.step()
        engine.save(weights)
        if checkouts != None:
            os.popen('cp %s %s' % (weights, checkout_format % i))
        for k in ['train', 'test']:
            if to_skip_evaluation[k]:
                continue
            print('Evaluating %sing data.' % k)
            tengine.evaluate(engine._model, data_loader[k], device = engine._device, print_freq = print_frequency)
    print('Training done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training Mask R-CNN model.')
    parser.add_argument('--dataset_type', type = str,
        help = 'The type of the dataset. It should be: cityscapes, label_me.')
    parser.add_argument('--train_image', type = str,
        help = 'The root of images of the training dataset.')
    parser.add_argument('--train_mask', type = str,
        help = 'The root of masks of the training dataset.')
    parser.add_argument('--test_image', type = str,
        help = 'The root of images of the testing dataset.')
    parser.add_argument('--test_mask', type = str,
        help = 'The root of masks of the testing dataset.')
    parser.add_argument('--image_extension', type = str, default = 'png',
        help = 'The file extension of the images in dataset. It will be useless if the dataset_type is cityscapes.')
    parser.add_argument('--class', type = str, default = 'data/classes/data.class',
        help = 'The root of the class names.')
    parser.add_argument('--crowd', type = str, default = None,
        help = 'The root of the crowded attributes.')
    parser.add_argument('--weights', type = str, default = None,
        help = 'The root of the weights.')
    parser.add_argument('--checkouts', type = str, default = None,
        help = 'The root of the weight checkouts.')
    parser.add_argument('--epoch', type = int, default = 100,
        help = 'The epoch for training.')
    parser.add_argument('--device', type = str, default = 'cuda',
        help = 'Choose the device to use.')
    parser.add_argument('--resized_width', type = int, default = 800,
        help = 'Resize each data by the specific width.')
    parser.add_argument('--resized_height', type = int, default = 450,
        help = 'Resize each data by the specific height.')
    parser.add_argument('--batch_size', type = int, default = 1,
        help = 'The size of each batch.')
    parser.add_argument('--learning_rate', type = float, default = 0.005,
        help = 'Learning rate.')
    parser.add_argument('--print_frequency', type = int, default = 100,
        help = 'Learning rate.')
    parser.add_argument('--to_skip_training_evaluation', dest = 'to_skip_training_evaluation', action = 'store_true',
        help = 'To skip evaluation of the training data for saving time.')
    parser.set_defaults(to_skip_training_evaluation = False)
    parser.add_argument('--to_skip_testing_evaluation', dest = 'to_skip_testing_evaluation', action = 'store_true',
        help = 'To skip evaluation of the testing data for saving time.')
    parser.set_defaults(to_skip_testing_evaluation = False)
    args = vars(parser.parse_args())
    main(
        dataset_type = args['dataset_type'],
        root_dataset = {
            'train_image': args['train_image'],
            'train_mask': args['train_mask'],
            'test_image': args['test_image'],
            'test_mask': args['test_mask'],
            'image_extension': None if args['dataset_type'] == 'cityscapes' else args['image_extension']
        },
        clazz = args['class'],
        crowd = args['crowd'],
        weights = args['weights'],
        checkouts = args['checkouts'],
        epoch = args['epoch'],
        device = args['device'],
        resized_size = (args['resized_height'], args['resized_width']),
        batch_size = args['batch_size'],
        learning_rate = args['learning_rate'],
        print_frequency = args['print_frequency'],
        to_skip_evaluation = {
            'train': args['to_skip_training_evaluation'],
            'test': args['to_skip_testing_evaluation']
        }
    )
