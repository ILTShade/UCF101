#-*-coding:utf-8-*-
'''
@FileName:
    Networks.py
@Description:
    Construct network
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2020/04/23 23:08
'''
import os
import math
import torch
from torch import nn
import numpy as np

# bottleneck
class Bottleneck(nn.Module):
    '''
    bottleneck part
    '''
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size = 3, stride = stride,
            padding = 1, bias = False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck.expansion,
            kernel_size = 1, bias = False
        )
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        '''
        forward
        '''
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
# construct resnet by bottleneck
class ResNet(nn.Module):
    '''
    resnet
    '''
    def __init__(self, block, layers, in_channels = 3, out_classes = 101):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1_custom = nn.Conv2d(
            in_channels, self.inplanes,
            kernel_size = 7, stride = 2,
            padding = 3, bias = False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.fc_custom = nn.Linear(512 * block.expansion, out_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride = 1):
        '''
        make layer by blocks
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size = 1, stride = stride, bias = False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        '''
        forward
        '''
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc_custom(x)
        return out

def weight_transform(weight_dict, target_input_channel):
    '''
    load pretrain weight, transfer it to target input channel
    '''
    input_weight_name = 'conv1_custom.weight'
    origin_input_channel = weight_dict[input_weight_name].shape[1]
    if origin_input_channel == target_input_channel:
        return weight_dict
    # cross modality
    transform_weight = weight_dict[input_weight_name].numpy()
    transform_weight = np.mean(transform_weight, axis = 1, keepdims = True)
    transform_weight = np.repeat(transform_weight, target_input_channel, axis = 1)
    weight_dict[input_weight_name] = torch.from_numpy(transform_weight)
    return weight_dict

def fc_transform(weight_dict, target_out_classes):
    '''
    load pretrain weight, transfer fc
    '''
    input_fc_name_list = ['fc_custom.weight', 'fc_custom.bias']
    for input_fc_name in input_fc_name_list:
        if weight_dict[input_fc_name].shape[0] < target_out_classes:
            raise Exception('out of range')
        weight_dict[input_fc_name] = weight_dict[input_fc_name][:target_out_classes,...]
    return weight_dict

class Network(nn.Module):
    '''
    network class
    '''
    def __init__(self, in_channels = 3, out_classes = 101, pretrain_path = None):
        super(Network, self).__init__()
        # construct network
        self.net = ResNet(Bottleneck, [3, 4, 23, 3], in_channels, out_classes)
        # check for path
        if pretrain_path is not None:
            if not os.path.exists(pretrain_path):
                raise Exception(f'{pretrain_path} does NOT exist')
            weight_dict = torch.load(pretrain_path, map_location = lambda storage, loc: storage)
            self.net.load_state_dict(fc_transform(weight_transform(weight_dict, in_channels), out_classes))
    def forward(self, x):
        '''
        forward
        '''
        return self.net(x)
    def save_pth(self, save_name):
        '''
        save param
        '''
        torch.save(self.net.state_dict(), save_name)

if __name__ == '__main__':
    net = Network(in_channels = 13, out_classes = 101, pretrain_path = './zoo/spatial_pretrain.pth')
    print(net)
