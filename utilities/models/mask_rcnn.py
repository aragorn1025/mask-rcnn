import torch
import torchvision

class MaskRCNN(torch.nn.Module):
    def __init__(self, number_classes, number_hidden_layer = 256):
        super(MaskRCNN, self).__init__()
        self._model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        box_predictor_features = self._model.roi_heads.box_predictor.cls_score.in_features
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(box_predictor_features, number_classes)
        self._model.roi_heads.box_predictor = box_predictor
        mask_predictor_features = self._model.roi_heads.mask_predictor.conv5_mask.in_channels
        mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(mask_predictor_features,
                                                                                  number_hidden_layer,
                                                                                  number_classes)
        self._model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(mask_predictor_features,
                                                                                                        number_hidden_layer,
                                                                                                        number_classes)

    def forward(self, images, targets=None):
        return self._model(images, targets)