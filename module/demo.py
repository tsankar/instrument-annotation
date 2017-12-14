import os
from detection import SSDDetect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
import skimage.io
import itertools

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

VID_NAME = 'clapton.mkv'
VID_PATH = os.path.expanduser('~/Videos/' + VID_NAME)
start = 64
length = 5

WEIGHTS = 'data/frozen_aug.caffemodel'
DEPLOY = 'data/deploy_finetune.prototxt'
LABELMAP = 'data/ILSVRC2016/labelmap_ilsvrc_finetune.prototxt'

print 'Running detection...'
ssd = SSDDetect(DEPLOY, WEIGHTS, LABELMAP)
# You may have to lower this value to get good results
conf_thresh = 0.3
boxgen = ssd.detect_vid(VID_PATH, start, length, conf_thresh)

colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

bb_frames = []
for batch, frame_boxes in boxgen:
    for frame, boxes in zip(batch, frame_boxes):
        plt.imshow(frame)
        curr_axis = plt.gca()
        for coords, score, label, label_name in boxes:
            color = colors[label]
            curr_axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            display_txt = '%s: %.2f'%(label_name, score)
            curr_axis.text(coords[0][0], coords[0][1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        fig = plt.gcf()
        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        bb_frames.append(data)
        plt.clf()

plt.close()

bb_frames = np.array(bb_frames)
print bb_frames.shape
skvideo.io.vwrite(os.path.join('data', VID_NAME.split('.')[0] + '.mp4'), bb_frames)
print 'Output written to ' + os.path.join('data', VID_NAME.split('.')[0] + '.mp4')
