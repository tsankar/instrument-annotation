import os
import matplotlib.pyplot as plt
import numpy as np

metric = 'loss'

if metric == 'acc':
    keyword = 'detection_eval ='
    offset = 17
elif metric == 'loss':
    keyword = ' loss ='
    offset = 8

f = open('/home/tharun/Downloads/frozen_aug.log', 'r')
vals = []
for line in f.readlines():
    i = line.find(keyword)
    if i != -1:
        vals.append(float(line[i + offset:]))

vals = np.array(vals)
if metric == 'acc':
    plt.plot(np.arange(1, len(vals) + 1, dtype='uint16'), vals, '-o')
    plt.title('mAP per Epoch for Frozen Layers with Data Augmentation')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.ylim((0, 1))
    plt.xticks(np.arange(len(vals) + 1))
elif metric == 'loss':
    x_axis = np.arange(len(vals), dtype='uint16')
    print x_axis
    plt.plot([x*10 for x in x_axis], vals, '-o')
    plt.title('Loss per Iteration for Frozen Layers with Data Augmentation')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

plt.show()
