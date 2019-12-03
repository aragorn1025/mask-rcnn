import torch
import torch.utils.data

class LabelMeDataset(torch.utils.data.Dataset):
    def __init__(self):
		super(LabelMeDataset, self).__init__()
        self._data = []
        raise NotImplementedError('LabelMeDataset.__init__() is not implement yet.')

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return torch.tensor(self._data[index][0]), torch.tensor(self._data[index][1])
