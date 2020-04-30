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
            '01',
            'spatial_dmd',
            25,
            8,
        )
        self.train_loader = self.dataset.get_loader('train')
        self.test_loader = self.dataset.get_loader('test')
        # networks
        self.net = Networks.Network(pretrain_path = './zoo/spatial_pretrain.pth')
        # device
        self.device = torch.device('cuda:0')
    def evaluate(self):
        '''
        evaluate network
        '''
        self.net.eval()
        self.net.to(self.device)
        # record result
        video_array = list()
        output_array = list()
        label_array = list()
        for _, video_name, image, label in tqdm(self.test_loader):
            image = image[:,0:3,:,:]
            image = image.to(self.device)
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
        print(f'{correct_count}/{total_count}')

if __name__ == '__main__':
    trainer = Trainer()
    trainer.evaluate()
