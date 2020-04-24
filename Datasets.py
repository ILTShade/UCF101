#-*-coding:utf-8-*-
'''
@FileName:
    Datasets.py
@Description:
    dataset for UCF101
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2020/04/24 00:04
'''
import os
import math
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils import data as Data
import torchvision.transforms as Transforms
from utils import dmd_transform

class SpatialDataset(Data.Dataset):
    '''
    spatial dataset
    '''
    def __init__(self, ucf_data_path, ucf_class_path, ucf_list_path, mode, transform):
        super(SpatialDataset, self).__init__()
        self.ucf_data_path = ucf_data_path
        self.mode = mode
        self.transform = transform
        self.window_length = 10
        self.test_count = 19
        self.train_slice = 3
        # read class label to name
        self.map_name_label = dict()
        with open(ucf_class_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if not len(line) == 2:
                raise Exception(f'there can be only two part for label and name')
            self.map_name_label[line[1]] = int(line[0]) - 1
        # read list, treat train and test as the same
        self.record_list = list()
        with open(ucf_list_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()[0]
            # split label, name, and so on
            type_name = line.split('/')[0]
            video_name = line.split('/', 1)[1].split('.', 1)[0]
            # frame count
            video_path = os.path.join(self.ucf_data_path, video_name)
            frame_count = len(list(filter(lambda x: x.endswith('.jpg'), os.listdir(video_path))))
            frame_count = frame_count - self.window_length + 1
            self.record_list.append((type_name, video_name, video_path, frame_count))
        # dmd_transformer
        self.output_root_path = os.path.join(os.path.dirname(self.ucf_data_path), 'npys_256')
        if not os.path.exists(self.output_root_path):
            os.mkdir(self.output_root_path)
        self.dmd_transfromer = dmd_transform.DMDTransform(self.window_length, mode = 'gpu')
    def __len__(self):
        if self.mode == 'train':
            return len(self.record_list)
        if self.mode == 'test':
            return len(self.record_list) * self.test_count
        raise Exception(f'does NOT support {self.mode}')
    def __getitem__(self, index):
        if self.mode == 'train':
            type_name, video_name, video_path, frame_count = self.record_list[index]
            data = list()
            for i in range(self.train_slice):
                low_bound = math.floor(i * frame_count / self.train_slice)
                high_bound = min(
                    math.ceil((i + 1) * frame_count / self.train_slice),
                    frame_count - 1
                )
                selected_index = random.randint(low_bound, high_bound)
                data.append(self.load_ucf_image(video_path, selected_index))
            sample = (type_name, video_name, data, self.map_name_label[type_name])
        elif self.mode == 'test':
            # load video
            selected_video = index // self.test_count
            type_name, video_name, video_path, frame_count = self.record_list[selected_video]
            # load index
            interval = int(frame_count / self.test_count)
            selected_index = (index % self.test_count) * interval
            data = self.load_ucf_image(video_path, selected_index)
            sample = (type_name, video_name, data, self.map_name_label[type_name])
        else:
            raise Exception(f'does NOT support {self.mode}')
        return sample
    def load_ucf_image(self, video_path, selected_index):
        '''
        load selected image
        '''
        image_name = os.path.join(video_path, f'frame{selected_index + 1:06d}.jpg')
        img = Image.open(image_name)
        return self.transform(img)
    def generate_dct_npy(self):
        '''
        generate dct npy file
        '''
        for type_name, video_name, video_path, frame_count in tqdm(self.record_list):
            output_video_path = os.path.join(self.output_root_path, f'{video_name}.npz')
            if os.path.exists(output_video_path):
                continue
            # load gray image
            files = filter(lambda f: f.endswith('.jpg'), os.listdir(video_path))
            files = sorted(files, key = lambda f: int(os.path.splitext(f)[0][len('frame'):]))
            # read image array
            image_array = [np.expand_dims(
                np.array(Image.open(os.path.join(video_path, f)).convert('L')),
                axis = -1,
                ) for f in files]
            image_array = [im.astype(np.float32) * 2. / 255. - 1 for im in image_array]
            image_array = np.stack(image_array, axis = 0)
            # transfer
            output_array, consume_time = self.dmd_transfromer.transfer(image_array)
            output_array = output_array * 128. / (2 * self.window_length)
            output_array = np.clip(output_array, -128, 127).astype(np.int8)
            if not output_array.shape[0] == frame_count:
                raise Exception(f'output shape mismatch')
            # save file
            np.savez_compressed(output_video_path, output_array)

class Dataset(object):
    '''
    dataset
    '''
    def __init__(self, ucf_data_path, ucf_anno_path, ucf_flag, batch_size, num_workers):
        if not ucf_flag in ['01', '02', '03']:
            raise Exception(f'out of range')
        self.train_dataset = SpatialDataset(
            ucf_data_path,
            os.path.join(ucf_anno_path, 'classInd.txt'),
            os.path.join(ucf_anno_path, f'trainlist{ucf_flag}.txt'),
            'train',
            Transforms.Compose([
                Transforms.RandomCrop(224),
                Transforms.RandomHorizontalFlip(),
                Transforms.ToTensor(),
                Transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            ]),
        )
        self.test_dataset = SpatialDataset(
            ucf_data_path,
            os.path.join(ucf_anno_path, 'classInd.txt'),
            os.path.join(ucf_anno_path, f'testlist{ucf_flag}.txt'),
            'test',
            Transforms.Compose([
                Transforms.Resize([224, 224]),
                Transforms.ToTensor(),
                Transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            ]),
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
    def get_loader(self, phase):
        '''
        data loader
        '''
        if phase == 'train':
            return Data.DataLoader(
                dataset = self.train_dataset,
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                drop_last = True,
            )
        if phase == 'test':
            return Data.DataLoader(
                dataset = self.test_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
            )
        raise Exception(f'does NOT support {phase}')
    def generate_dct_npy(self):
        '''
        generate dct npy file
        '''
        self.train_dataset.generate_dct_npy()
        self.test_dataset.generate_dct_npy()

if __name__ == '__main__':
    dataset = Dataset(
        '/datasets/UCF101/jpegs_256',
        '/datasets/UCF101/UCF_list',
        '01',
        25,
        8,
    )
    train_loader = dataset.get_loader('train')
    test_loader = dataset.get_loader('test')
    print(len(train_loader))
    print(len(test_loader))
    dataset.generate_dct_npy()
