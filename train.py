import argparse
import os
import PIL
import torch
import torchvision

import thirdparty.torchvision.detection.utils
import thirdparty.torchvision.detection.engine
import utilities.datasets as udatasets
import utilities.engine as uengine
import utilities.models as umodels
import utilities.tools as utools

def _get_dataset(dataset_type, root_images, root_masks, image_extension = None, transforms_images = None, transforms_masks = None):
    if dataset_type == 'cityscapes':
        return udatasets.cityscapes_dataset.CityscapesDataset(root_images, root_masks, transforms_images, transforms_masks)
    if dataset_type == 'label_me':
        return udatasets.label_me_dataset.LabelMeDataset(root_images, root_masks, image_extension, transforms_images, transforms_target)
    raise NotImplementedError('Unknown dataset type.')

def main(dataset_type, root_dataset, classes, weights, epoch, resized_size, batch_size, learning_rate, to_use_gpu):
    for key in ['train_image', 'train_mask', 'test_image', 'test_mask']:
        utools.file.check_directory(key, root_dataset[key])
    utools.file.check_file('weights', weights)
    
    dataset = {}
    data_loader = {}
    transforms = utools.instance_segmentation.get_transforms(resized_size)
    for key in ['train', 'test']:
        dataset[key] = _get_dataset(
            dataset_type,
            root_dataset['%s_image' % key],
            root_dataset['%s_mask' % key],
            root_dataset['image_extension'],
            transforms_images = transforms['images'],
            transforms_masks = transforms['masks']
        )
        data_loader[key] = torch.utils.data.DataLoader(
            dataset[key],
            batch_size = batch_size,
            shuffle = (key == 'train'),
            num_workers = 1,
            collate_fn = thirdparty.torchvision.detection.utils.collate_fn
        )
    
    class_names = utools.general.get_classes(classes)
    model = umodels.mask_rcnn.MaskRCNN(number_classes = len(class_names))
    parameters = [p for p in model.parameters() if p.requires_grad]
    engine = uengine.Engine(
        model = model,
        criterion = None,
        optimizer = torch.optim.SGD(parameters, lr = learning_rate, momentum = 0.9, weight_decay = 0.0005),
        device = to_use_gpu
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(engine._optimizer, step_size = 3, gamma = 0.1)
    if os.path.isfile(weights):
        engine.load(weights)
    weights_directory = os.path.abspath(os.path.join(weights, '..'))
    if not os.path.isdir(weights_directory):
        os.makedirs(weights_directory, exist_ok = True)
    
    for i in range(0, epoch):
        thirdparty.torchvision.detection.engine.train_one_epoch(engine._model, engine._optimizer, data_loader['train'], engine._device, i, print_freq = 10)
        lr_scheduler.step()
        thirdparty.torchvision.detection.engine.evaluate(engine._model, data_loader['test'], device = engine._device)
        engine.save(weights)
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
    parser.add_argument('--classes', type = str, default = 'classes/class.names',
        help = 'The root of the class names.')
    parser.add_argument('--weights', type = str, default = None,
        help = 'The root of the weights.')
    parser.add_argument('--epoch', type = int, default = 100,
        help = 'The epoch for training.')
    parser.add_argument('--resized_width', type = int, default = 800,
        help = 'Resize each data by the specific width.')
    parser.add_argument('--resized_height', type = int, default = 400,
        help = 'Resize each data by the specific height.')
    parser.add_argument('--batch_size', type = int, default = 1,
        help = 'The size of each batch.')
    parser.add_argument('--learning_rate', type = float, default = 0.005,
        help = 'Learning rate.')
    parser.add_argument('--to_use_gpu', dest = 'to_use_gpu', action = 'store_true',
        help = 'To use GPU if available.')
    parser.add_argument('--to_use_cpu', dest = 'to_use_gpu', action = 'store_false',
        help = 'To use CPU rather than GPU.')
    parser.set_defaults(to_use_gpu = True)
    args = parser.parse_args()
    
    root_dataset = {
        'train_image': args.train_image,
        'train_mask': args.train_mask,
        'test_image': args.test_image,
        'test_mask': args.test_mask,
        'image_extension': None if args.dataset_type == 'cityscapes' else args.image_extension
    }
    resized_size = (args.resized_height, args.resized_width)
    main(args.dataset_type, root_dataset, args.classes, args.weights, args.epoch, resized_size, args.batch_size, args.learning_rate, args.to_use_gpu)
