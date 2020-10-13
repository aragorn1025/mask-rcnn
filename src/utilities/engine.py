import torch
import torchvision

from .tools.system import *

class Engine:
    def __init__(self, model, criterion = None, optimizer = None, device = None):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = Engine.get_device(device)
        self._model.to(self._device)
    
    def do(self, action, data_loader):
        if action == 'train':
            return self._train(data_loader)
        if action == 'test':
            return self._test(data_loader)
        raise ValueError('Action should be \'train\' or \'test\', but action is \'%s\'' % action)
    
    def get_outputs(self, inputs, is_tensor = False):
        """
        Get the outputs through the model for the inputs.
        
        Keyword arguments:
            inputs    -- The inputs.
            is_tensor -- Set True if The inputs is tensor.
        Return:
            the outputs through the model for the inputs.
        """
        if not is_tensor:
            inputs = torchvision.transforms.ToTensor()(inputs)
        inputs = inputs.to(self._device)
        self._model.eval()
        return self._model([inputs])
    
    def load(self, weight_path):
        self._model.load_state_dict(torch.load(weight_path))
        self._model.eval()
    
    def save(self, weight_path):
        torch.save(self._model.state_dict(), weight_path)
    
    def _backward(self, loss):
        loss.backward()
        self._optimizer.step()
    
    def _forward(self, inputs):
        inputs = inputs.to(self._device)
        self._model.eval()
        return self._model(inputs)
    
    def _get_loss(self, outputs, targets):
        targets = targets.to(self._device)
        loss = self._criterion(outputs, targets)
        return loss
    
    def _test(self, data_loader, to_get_execution_time = False):
        if to_get_execution_time:
            return tools.system.get_execution_time(self._test, data_loader)
        self._model.eval()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(data_loader):
            outputs = self._forward(inputs)
            loss = self._get_loss(outputs, targets)
            running_loss += loss.data.item()
        return running_loss / len(data_loader)
    
    def _train(self, data_loader, to_get_execution_time = False):
        if to_get_execution_time:
            return tools.system.get_execution_time(self._train, data_loader)
        self._model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(data_loader):
            self._optimizer.zero_grad()
            outputs = self._forward(inputs)
            loss = self._get_loss(outputs, targets)
            running_loss += loss.data.item()
            self._backward(loss)
        return running_loss / len(data_loader)
    
    def get_device(device = None):
        if device == None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == True:
            return Engine.get_device()
        if type(device) is torch.device:
            return Engine.get_device(str(device))
        if type(device) is not str:
            return torch.device('cpu')
        device = device.lower().replace('cpu:0', 'cpu').replace('gpu', 'cuda')
        if device == 'cuda':
            return Engine.get_device()
        if len(device) <= 5 or device[:5] != 'cuda:':
            return torch.device('cpu')
        try:
            device_id = int(device.replace('cuda:', ''))
            device_name = torch.cuda.get_device_name(device_id)
            return torch.device('cuda:%d' % device_id)
        except AssertionError:
            return torch.device('cpu')
        except ValueError:
            return torch.device('cpu')
