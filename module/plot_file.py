import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
import skvideo.io
from ast import literal_eval as make_tuple

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

VID_NAME = 'clapton.mkv'
VID_PATH = os.path.expanduser('~/Videos/' + VID_NAME)

FRAME = 320

f = open(VID_NAME + '_result.txt', 'r')
lines = f.readlines()

colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

vidgen = skvideo.io.vreader(VID_PATH)

i = 0
for frame, line in zip(vidgen, lines):
    if i == FRAME:
        boxes = []
        k = 0
        j = line.find('(', k)
        while(j != -1):
            k = line.find(') ', j)
            boxes.append(make_tuple(line[j:k+1]))
            begin = k
            j = line.find('(', k)

        plt.imshow(frame)
        curr_axis = plt.gca()
        for coords, score, label, label_name in boxes:
            color = colors[label]
            curr_axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            display_txt = '%s: %.2f'%(label_name, score)
            curr_axis.text(coords[0][0], coords[0][1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        fig = plt.gcf()
        fig.canvas.draw()

        plt.show()

        break
    i += 1
