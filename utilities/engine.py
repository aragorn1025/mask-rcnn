import re
import time
import torch
import torchvision

class Engine:
    def __init__(self, model, criterion = None, optimizer = None, device = None):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = Engine.get_device(device)
        self._model.to(self._device)

    def get_outputs(self, inputs, is_tensor = False):
        if not is_tensor:
            inputs = torchvision.transforms.ToTensor()(inputs)
        inputs = inputs.to(self._device)
        self._model.eval()
        return self._model([inputs])

    def do(self, action, data_loader):
        if action == 'train':
            return self._train(data_loader)
        if action == 'test':
            return self._test(data_loader)
        raise ValueError('Action should be \'train\' or \'test\', but action is \'%s\'' % action)

    def load(self, weight_path):
        self._model.load_state_dict(torch.load(weight_path))
        self._model.eval()

    def save(self, weight_path):
        torch.save(self._model.state_dict(), weight_path)

    def _train(self, data_loader):
        time_start = time.time()
        self._model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(data_loader):
            self._optimizer.zero_grad()
            outputs = self._forward(inputs)
            loss = self._get_loss(outputs, targets)
            running_loss += loss.data.item()
            self._backward(loss)
        time_end = time.time()
        return (time_end - time_start, running_loss / len(data_loader))

    def _test(self, data_loader):
        time_start = time.time()
        self._model.eval()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(data_loader):
            outputs = self._forward(inputs)
            loss = self._get_loss(outputs, targets)
            running_loss += loss.data.item()
        time_end = time.time()
        return (time_end - time_start, running_loss / len(data_loader))

    def _forward(self, inputs):
        inputs = inputs.to(self._device)
        self._model.eval()
        return self._model(inputs)

    def _get_loss(self, outputs, targets):
        targets = targets.to(self._device)
        loss = self._criterion(outputs, targets)
        return loss

    def _backward(self, loss):
        loss.backward()
        self._optimizer.step()

    def get_device(device = None):
        if device == None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == True:
            return Engine.get_device()
        if type(device) is not str:
            return torch.device('cpu')
        device = device.lower()
        if device in ['gpu', 'cuda']:
            return Engine.get_device()
        if not re.compile('cuda*').match(device):
            return torch.device('cpu')
        try:
            device_id = int(device.replace('cuda:', ''))
            device_name = torch.cuda.get_device_name(device_id)
            return torch.device('cuda:%d' % device_id)
        except AssertionError:
            return torch.device('cpu')
        except ValueError:
            return torch.device('cpu')