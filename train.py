import argparse
import os
import torch

import utilities
from utilities.datasets.LabelMeDataset import LabelMeDataset
from utilities.models.MaskRCNN import MaskRCNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the linear regression model.')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
        help='The size of each batch')
    parser.add_argument('-e', '--epoch', type=int, default=100,
        help='The epoch for training')
    parser.add_argument('--weights', type=str, default=None,
        help='The weights to loaded')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
        help='Learning rate')
    parser.add_argument('--data_directory', type=str, default='./data/dataset_b',
        help='The root directory of data')
    args = vars(parser.parse_args())

    root_directory = args['data_directory']
    model = MaskRCNN()
    engine = utilities.Engine(
        model = model,
        criterion = torch.nn.MSELoss(),
        optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])
    )
    dataset_train = LabelMeDataset()
    dataset_testt = LabelMeDataset()
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    data_loader_testt = torch.utils.data.DataLoader(dataset_testt, batch_size=args['batch_size'], shuffle=False, num_workers=4)
    if args['weights'] != None and os.path.isfile(args['weights']):
        engine.load(args['weights'])
    if not os.path.isdir('weights'):
        os.mkdir('weights')
	for i in range(0, args['epoch']):
        _, loss_train = engine.do('train', data_loader_train)
        _, loss_testt = engine.do('test', data_loader_testt)
        print('Epoch [%d]: %.8f, %.8f' % (i, loss_train, loss_testt))
	    engine.save(os.path.join('weights', 'weights.pth'))
    print('Training done.')
