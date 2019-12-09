import torch
import scipy
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def self_define_model(num_classes,
                      backbone,
                      backbone_out_channels,
                      anchor_size,
                      aspect_ratios,
                      name_featmap,
                      output_size,
                      sampling_ratio):
    backbone = backbone.features
    backbone.out_channels = backbone_out_channels
    anchor_generator = AnchorGenerator(sizes=(anchor_size,),
                                       aspect_ratios=(aspect_ratios,))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=name_featmap,
                                                    output_size=output_size,
                                                    sampling_ratio=sampling_ratio)
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model
    
def maskrcnn_resnet50_fpn(num_classes, hidden_layer, pretrained=True):
    if pretrained is False:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model