import glob
import os
import PIL
import torch
import torchvision

class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_root,
                 masks_root = None,
                 images_file_name = '*',
                 masks_file_name = '*',
                 transforms = None,
                 transforms_target = None):
        super(MaskRCNNDataset, self).__init__()
        self._images = sorted(glob.glob(os.path.join(images_root, images_file_name)))
        self._masks = sorted(glob.glob(os.path.join(masks_root, masks_file_name))) if masks_root else []
        if len(self._masks) == 0:
            print('The dataset is for testing only.')
        elif len(self._images) != len(self._masks):
            raise IOError('The number of images (%d) and masks (%d) are not same.' % (len(self._images), len(self._masks)))
        self._transforms = transforms if transforms else torchvision.transforms.ToTensor()
        self._transforms_target = transforms_target

    def __getitem__(self, index):
        image = PIL.Image.open(self._images[index]).convert("RGB")
        image = self._transforms(image)

        if len(self._masks) == 0:
            target = {}
            return image, target

        target = self._get_target(mask)
        target["image_id"] = torch.tensor([index])

        return image, target

    def _get_target(self, index):
        raise NotImplementedError('MaskRCNNDataset._get_target(self, mask) is not implement yet.')

    def __len__(self):
        return len(self._images)