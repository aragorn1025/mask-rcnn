import base64
import io
import numpy as np
import PIL.Image
import PIL.ImageDraw
import math

import torch
import torch.utils.data
import PIL
import json

class LabelMeDataset(MaskRCNNDataset):
    
    def __init__(self, images_root, 
                 image_extension, 
                 masks_root, 
                 images_file_name = '*',
                 masks_file_name = '*', 
                 transforms = None, 
                 transforms_target = None):        
        super(LabelMeDataset, self).__init__(images_root,
                                             masks_root,
                                             "*.%s"% image_extension,
                                             "*.json",
                                             transforms,
                                             transforms_target)


    def _get_target(self, index):
        data = json.load(open(self._masks[index]))       
        image_json = LabelMeDataset._img_b64_to_arr(data['imageData'])
        label, label_names = LabelMeDataset._labelme_shapes_to_label(image_json.shape, data['shapes'])
        mask = PIL.Image.fromarray(label)
        labels=[]
        for label_name in label_names:
            labels.append(label_name)
                

        if self._transforms_target :
            mask = self._transforms_target(mask)
            
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

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

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
                    

        return target
    
    def _img_b64_to_arr(img_b64):
        f = io.BytesIO()
        f.write(base64.b64decode(img_b64))
        img_arr = np.array(PIL.Image.open(f))
        return img_arr
    
    def _shape_to_mask(img_shape, points, shape_type=None,
                      line_width=10, point_size=5):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
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
    
    def _shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
        assert type in ['class', 'instance']
        
        cls = np.zeros(img_shape[:2], dtype=np.int32)
        if type == 'instance':
            ins = np.zeros(img_shape[:2], dtype=np.int32)
            instance_names = ['_background_']
        value = {'_background_': 0}
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            value_value = len(value)
            shape_type = shape.get('shape_type', None)
            if type == 'class':
                cls_name = str(int(label)*1000 + len(value))
                value[cls_name] = value_value
            elif type == 'instance':
                cls_name = label.split('-')[0]
                if label not in instance_names:
                    instance_names.append(label)
                ins_id = instance_names.index(label)
            cls_id = label_name_to_value[cls_name]
            mask = LabelMeDataset._shape_to_mask(img_shape[:2], points, shape_type)
            cls[mask] = cls_id
            
            if type == 'instance':
                ins[mask] = ins_id
                
        if type == 'instance':
            return cls, ins
        return cls
    
    def _labelme_shapes_to_label(img_shape, shapes):
        
        label_name_to_value = {'_background_': 0}
        for shape in shapes:
            label_name = shape['label'] 
            label_value = len(label_name_to_value)
            label_name=str(int(label_name)*1000+len(label_name_to_value))
            label_name_to_value[label_name] = label_value
                
        lbl = LabelMeDataset._shapes_to_label(img_shape, shapes, label_name_to_value)
        return lbl, label_name_to_value