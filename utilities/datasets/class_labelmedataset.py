# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:57:51 2019

@author: Jia-Rou
"""

import os
import numpy as np
import torch
import torch.utils.data
import PIL
from torchvision import transforms

import json
import PIL.Image
from labelme import utils
    
class LabelmeDataset(torch.utils.data.Dataset):
    def __init__(self, root, resize, cropsize):
        self._root = root
        self._resize = resize
        self._cropsize = cropsize

        self._json = list(sorted(os.listdir(os.path.join(root))))


    def __getitem__(self, idx):
        json_path = os.path.join(self._root, self._json[idx])
        if os.path.isfile(json_path):
            data = json.load(open(json_path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
            img = PIL.Image.fromarray(img)
            mask = PIL.Image.fromarray(lbl)
            labels=[]
            for lbl_name in lbl_names:
                labels.append(lbl_name)
        

        transform = transforms.Compose([transforms.Resize(self._resize),
                                        transforms.CenterCrop(self._cropsize),
                                        transforms.ToTensor(),
                                        ])
        img = transform(img)
#        print(img)    

        transforms_target = transforms.Compose([transforms.Resize(self._resize, PIL.Image.NEAREST),
                                                transforms.CenterCrop(self._cropsize),
                                                ])
        mask = transforms_target(mask)

        mask = np.array(mask)
#        print (mask)
        
        obj_ids = np.unique(mask)
#        print(obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks

        masks = mask == obj_ids[:, None, None]
#        print (masks)
#        print(masks.size)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])            
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = labels[1:]
        labels = list(map(int, labels))
        labels = np.array(labels)
        labels = torch.tensor(labels/1000, dtype=torch.int64)
#        print(labels)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
#        print(masks)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
                    

        return img, target

    def __len__(self):
        return len(self._imgs)
    
    





