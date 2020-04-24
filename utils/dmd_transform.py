#-*-coding:utf-8-*-
'''
@FileName:
    dmd_transform.py
@Description:
    for video file, transfor it to npy file
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2020/04/07 18:26
'''
import time
import math

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from scipy.fftpack import dct as DCT

CUDA_FUNCTION = '''
    __global__ void STDCT(float *output, float *input, float *weight)
    {
        // THWC format
        int Tid = threadIdx.x;
        int Hid = blockIdx.x;
        int Wid = blockIdx.y;
        int Cid = blockIdx.z;
        // input index start and output index start and index step
        int input_index_start = Tid * gridDim.x * gridDim.y * gridDim.z + \
            Hid * gridDim.y * gridDim.z + \
            Wid * gridDim.z + \
            Cid;
        int output_index_start = TEMPLATEN * Tid * gridDim.x * gridDim.y * gridDim.z + \
            Cid * gridDim.x * gridDim.y + \
            Hid * gridDim.y + \
            Wid;
        int index_step = gridDim.x * gridDim.y * gridDim.z;
        // traverse
        for (int i = 0; i < TEMPLATEN; i++) {
            output[output_index_start+index_step*i] = 0.f;
            for (int j = 0; j < TEMPLATEN; j++) {
                output[output_index_start+index_step*i] += \
                    (input[input_index_start+index_step*j] * weight[i*TEMPLATEN+j]);
            }
        }
    }
'''

class DMDTransform:
    '''
    Transform
    Input: image array
    Output: dct ndarray
    '''
    def __init__(self, dct_length, mode = 'gpu'):
        self.device = pycuda.autoinit.device
        self.dct_length = dct_length
        self.mode = mode
        # mode gpu
        if self.mode == 'gpu':
            # cuda function
            self.CUDA_FUNCTION_INS = CUDA_FUNCTION.replace('TEMPLATEN', str(dct_length))
            self.DCTGPU = SourceModule(self.CUDA_FUNCTION_INS).get_function('STDCT')
            # weights
            self.weights = np.zeros(shape = (dct_length, dct_length), dtype = np.float32)
            for index_i in range(dct_length):
                for index_j in range(dct_length):
                    self.weights[index_i,index_j] = 2*math.cos(math.pi*index_i*(2*index_j+1)/(2*dct_length))
    def transfer(self, image_array):
        '''
        transfer image array to dct array
        '''
        # image shape must be THWC
        if not len(image_array.shape) == 4 and image_array.shape[3] in [1, 3]:
            raise Exception(f'input image shape must be 4 dim, and last dim must be 1 or 3')
        if not np.max(np.abs(image_array)) <= 1:
            raise Exception(f'image array must be in range [-1, 1]')
        # input_length, vid_length
        input_length = image_array.shape[0]
        vid_length = input_length - self.dct_length + 1
        start_time = time.time()
        # transfer
        if self.mode == 'cpu':
            # now vid shape is THWC, [-1, 1], vid length is self.vid_pad
            # slide window
            array = [image_array[index:index+self.dct_length,...] for index in range(vid_length)]
            # dct transform
            array = [
                DCT(array[index], type = 2, n = None, axis = 0, norm = None, overwrite_x = False)
                for index in range(vid_length)
            ]
            # transpose and concat
            array = [
                array[index].transpose(0, 3, 1, 2).reshape(-1,array[index].shape[1],array[index].shape[2])
                for index in range(vid_length)
            ]
            # stack
            output = np.stack(array, axis = 0)
        elif self.mode == 'gpu':
            output_shape = [vid_length, self.dct_length * image_array.shape[3]] + list(image_array.shape[1:3])
            output = np.zeros(shape = output_shape, dtype = np.float32)
            # grid and block
            block = (vid_length, 1, 1)
            grid = image_array.shape[1:]
            self.DCTGPU(drv.Out(output), drv.In(image_array), drv.In(self.weights), block = block, grid = grid)
        else:
            raise NotImplementedError
        consume_time = time.time() - start_time
        return output, consume_time
