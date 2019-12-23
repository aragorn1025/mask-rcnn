import argparse
import torch
import numpy as np
import os
import PIL
import cv2
import torchvision
import matplotlib.pyplot as plt
import utilities.engine as engine
import utilities.models.mask_rcnn as mask_rcnn
import utilities.tools.instance_segmentation as instance_segmentation

def get_predictions(image, engine, threshold, category_names):
    device = torch.device('cpu')
    predictions = engine.get_outputs(image)[0]
    scores = list(predictions['scores'].detach().cpu().numpy())
    
    predictions_pass_threshold = [scores.index(x) for x in scores if x > threshold]
    if len(predictions_pass_threshold) == 0:
        return None, None, None
    predictions_pass_threshold_last_index = [scores.index(x) for x in scores if x > threshold][-1]
    
    masks = (predictions['masks'] > 0.5).squeeze().detach().cpu().numpy()
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions['boxes'].detach().cpu().numpy())]
    labels = [category_names[i] for i in list(predictions['labels'].cpu().numpy())]

    if predictions_pass_threshold_last_index > 0:
        masks = masks[:predictions_pass_threshold_last_index + 1]
    else:
        if len(masks.shape) == 2:
            masks = np.asarray([masks])
    boxes = boxes[:predictions_pass_threshold_last_index + 1]
    labels = labels[:predictions_pass_threshold_last_index + 1]    
    return masks, boxes, labels

def show_predictions(image, masks, boxes, labels):
    n = 0 if labels == None else len(labels)
    if n < 1:
        print("There is no prediction.")
    elif n == 1:
        print("There is 1 prediction.")
    else:
        print("There are %d predictions." % n)
    image = image.squeeze(0)
    image = image.mul(255).permute(1, 2, 0).byte().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rectangle_thickness = 2
    text_size = 1
    text_thickness = 2
    result = np.copy(image)
    for i in range(0, n):
        colored_mask = instance_segmentation.get_random_colored_mask(masks[i])
        result = cv2.addWeighted(result, 1, colored_mask, 0.5, 0)
        cv2.rectangle(result, boxes[i][0], boxes[i][1], color = (0, 255, 0), thickness = rectangle_thickness)
        cv2.putText(result, str(labels[i]), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,
                    text_size, (0,255,0), thickness = text_thickness)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(20,30))
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    result = PIL.Image.fromarray(result)
    return result
    
def PIL_to_tensor(image):
    loader = torchvision.transforms.ToTensor()
    image = loader(image).unsqueeze(0)
    return image.to(torch.device('cpu'), torch.float)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Instance detection using Mask R-CNN .')
    parser.add_argument('-f')
    parser.add_argument('--weights', type=str, default = './weights/mask-rcnn-cityscapes.pth',
        help='The weights to loaded')
    parser.add_argument('--input', type=str, default = './test/007c1782-671f-47e3-9cf1-98e4fb001367.mov-0001.jpg',
        help='The path of file')
    parser.add_argument('--output', type=str, default = './output',
        help='The path to save')
    args = vars(parser.parse_args())
    if args['weights'] == None:
        raise ValueError('Thw weights file path should be set.')
    if not os.path.isfile(args['weights']):
        raise IOError('The weights file is not found.')
    if args['input'] == None:
        raise ValueError('The input file path should be set.')
    if not os.path.isfile(args['input']):
        raise IOError('The inputs file is not found.')
    if args['output'] == None:
        raise ValueError('The outputs file path should be set.')
        
    engine = engine.Engine(model = mask_rcnn.MaskRCNN(number_classes = 11), device = torch.device('cpu'))
    #engine.load(args['weights'])

    image = PIL.Image.open(args['input'])
    image = PIL_to_tensor(image)

    CITYSCAPES_INSTANCE_CATEGORY_NAMES= [
        'background',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'caravan',
        'trailer',
        'train',
        'motorcycle',
        'bicycle']
    
    masks, boxes, labels = get_predictions(image, engine, 0.2, CITYSCAPES_INSTANCE_CATEGORY_NAMES)
    result = show_predictions(image, masks, boxes, labels)
    saving_path = args['output'] + '/' + args['input'].rstrip('.jpg').lstrip('./test') + 'prediction' +'.jpg'
    result.save(saving_path)