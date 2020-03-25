import numpy as np
import math

import torch
import PIL
import PIL.ImageDraw
import json
from .mask_rcnn_dataset import MaskRCNNDataset
import uuid

class LabelMeDataset(MaskRCNNDataset):
    '''
    Reference: https://github.com/wkentaro/labelme/blob/612b40df6ff5673dab8bc9b68dbd4d1fe17630ea/labelme/utils/shape.py
    '''
    
    def __init__(self, root_images, root_masks, image_extension, class_names = [], transforms = None, transforms_target = None):
        super(LabelMeDataset, self).__init__(
            root_images = root_images,
            root_masks = root_masks,
            file_name_images = "*.%s"% image_extension,
            file_name_masks = "*.json",
            transforms = transforms,
            transforms_target = transforms_target
        )
        self._class_names = class_names

    def _get_target(self, index):
        data = json.load(open(self._masks[index]))   
        image_shape = data['imageHeight'], data['imageWidth'], 3
        mask, label_names = self._labelme_shapes_to_label(image_shape, data['shapes'])
        mask = PIL.Image.fromarray(mask)
        if self._transforms_target :
            mask = self._transforms_target(mask)
        mask = np.array(mask)
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
    
    def _shape_to_mask(image_shape, points, shape_type=None, line_width=10, point_size=5):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)
        xy = [tuple(point) for point in points]
        if shape_type == 'circle':
            assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        elif shape_type == 'rectangle':
            assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
            draw.rectangle(xy, outline=1, fill=1)
        elif shape_type == 'line':
            assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == 'linestrip':
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == 'point':
            assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
            cx, cy = xy[0]
            r = point_size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
        else:
            assert len(xy) > 2, 'Polygon must have points more than 2'
            draw.polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask
    
    def _shapes_to_label(img_shape, shapes, label_name_to_value):
        cls = np.zeros(img_shape[:2], dtype=np.int32)
        ins = np.zeros_like(cls)
        instances = []
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            if group_id is None:
                group_id = uuid.uuid1()
            shape_type = shape.get('shape_type', None)
    
            cls_name = label
            instance = (cls_name, group_id)
    
            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]
    
            mask = LabelMeDataset._shape_to_mask(img_shape[:2], points, shape_type)
            cls[mask] = cls_id
            ins[mask] = ins_id
    
        return cls, ins
    
    def _merge_classes_and_instances(cls, ins, cls_multiplier = 1000):
        return cls * cls_multiplier + ins
    
    def _labelme_shapes_to_label(self, image_shape, shapes):
        label_name_to_value = {'background': 0}
        for shape in shapes:
            label_name = shape['label']
            if label_name in label_name_to_value.keys():
                continue
            if label_name in self._class_names:
                label_name_to_value[label_name] = self._class_names.index(label_name)
            else:
                label_name_to_value[label_name] = len(self._class_names) + len(label_name_to_value)
        cls, ins = LabelMeDataset._shapes_to_label(image_shape, shapes, label_name_to_value)
        lbl = LabelMeDataset._merge_classes_and_instances(cls, ins)
        return lbl, label_name_to_value
