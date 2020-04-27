#-*-coding:utf-8-*-
'''
@FileName:
    response_plot.py
@Description:
    plot response heatmap
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2020/04/10 17:39
'''
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# load file
npz_file_path = '/datasets/UCF101/npys_256'
files = filter(lambda f: f.endswith('.npz'), os.listdir(npz_file_path))
files = list(map(lambda f: os.path.join(npz_file_path, f), files))
selected_index = random.randint(0, len(files) - 1)
video_name = os.path.splitext(os.path.basename(files[selected_index]))[0]
# load array
output_array = np.load(files[selected_index])['arr_0']
output_array = np.abs(output_array)
selected_slice = random.randint(0, output_array.shape[0] - 1)
output = output_array[selected_slice,...]
# plot
window_length = output.shape[0]
plt.figure(0)
for i in range(window_length):
    plt.subplot(1, window_length, i + 1)
    plt.imshow(output[i])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.savefig(f'/home/sunhanbo/workspace/UCF101/record/dmd_result/{video_name}.png')
