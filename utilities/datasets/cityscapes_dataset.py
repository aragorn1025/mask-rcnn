import glob
import math
import numpy as np
import os
import PIL
import torch
import torchvision
from .mask_rcnn_dataset import MaskRCNNDataset

class CityscapesDataset(MaskRCNNDataset):
    def __init__(self,
                 images_root = "data/cityscapes/leftImg8bit/train/*",
                 masks_root = "data/cityscapes/gtFine/train/*",
                 transforms = None,
                 transforms_target = None):
        super(CityscapesDataset, self).__init__(images_root,
                                                masks_root,
                                                "*_leftImg8bit.png",
                                                "*_gtFine_instanceIds.png",
                                                transforms,
                                                transforms_target)

    def _get_target(self, mask):
        mask = np.array(mask)
        # reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        mask = np.where(mask < 24 * 1000, 0, mask)
        mask = np.where(mask >= 34 * 1000, 0, mask)
        mask = np.where(mask == 0, 0, mask - (24 - 1) * 1000)
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids > 0]
        masks = mask == object_ids[:, None, None]
        n_objects = len(object_ids)
        boxes = []
        for i in range(n_objects):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(object_ids / 1000, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except IndexError:
            area = []
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target