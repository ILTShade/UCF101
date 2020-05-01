#-*-coding:utf-8-*-
'''
@FileName:
    Trainers.py
@Description:
    train and evaluate network
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2020/04/24 02:16
'''
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import DataParallel
import Datasets
import Networks

class Trainer(object):
    '''
    trainer
    '''
    def __init__(self):
        # datasets
        self.dataset = Datasets.Dataset(
            '/datasets/UCF101/jpegs_256',
            '/datasets/UCF101/UCF_list',
            '04',
            'spatial_dmd',
            30,
            24,
        )
        self.train_loader = self.dataset.get_loader('train')
        self.test_loader = self.dataset.get_loader('test')
        # networks
        self.net = Networks.Network(in_channels = 13, out_classes = 10, pretrain_path = './zoo/spatial_pretrain.pth')
    def train(self):
        '''
        train network
        '''
        # set net on gpu
        self._load_net_device()
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [200, 300], gamma = 0.1)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.1,patience=10,verbose=True)
        # init test
        init_accuracy = self.evaluate()
        print(f'init accuracy is {init_accuracy}')
        # epochs
        for epoch in range(400):
            # train
            self.net.train()
            scheduler.step()
            tqdm_loader = tqdm(self.train_loader)
            for _, _, image, label in tqdm_loader:
                output = None
                for i in range(len(image)):
                    if i == 0:
                        output = self.net(image[i])
                    else:
                        output += self.net(image[i])
                label = label.to(output.device)
                # loss and backward
                loss = criterion(output, label)
                self.net.zero_grad()
                loss.backward()
                optimizer.step()
                tqdm_loader.set_description(f'loss is {loss.item():.4f}')
            # test and scheduler
            if epoch == 0 or (epoch + 1) % 10 == 0:
                accuracy = self.evaluate()
                print(f'epoch {epoch+1:04d}: accuracy is {accuracy}')
                torch.save(self.net.module.state_dict(), f'./zoo/spatial_dmd_{epoch+1}_{accuracy}.pth')
    def evaluate(self):
        '''
        evaluate network
        '''
        self._load_net_device()
        self.net.eval()
        # record result
        video_array = list()
        output_array = list()
        label_array = list()
        for _, video_name, image, label in tqdm(self.test_loader):
            output = self.net(image)
            # transfer to cpu
            video_array.extend(list(video_name))
            output_array.append(output.detach().cpu().numpy())
            label_array.append(label)
        # concatenate and reshape
        output_array = np.concatenate(output_array, axis = 0)
        label_array = np.concatenate(label_array, axis = 0)
        # statistics
        record_result = dict()
        for i, video_name in enumerate(video_array):
            if video_name not in record_result.keys():
                record_result[video_name] = [output_array[i], label_array[i]]
            else:
                if not record_result[video_name][1] == label_array[i]:
                    raise Exception(f'mismatch label')
                record_result[video_name][0] += output_array[i]
        # evaluate
        correct_count = 0
        total_count = 0
        for value in record_result.values():
            output = value[0]
            label = value[1]
            # calculate
            if np.argmax(output) == label:
                correct_count += 1
            total_count += 1
        return correct_count / total_count
    def _load_net_device(self):
        '''
        load net to device
        '''
        if not isinstance(self.net, DataParallel):
            self.net.to(torch.device('cuda:0'))
            self.net = DataParallel(self.net, device_ids = [0, 1])

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
