import os
import sys
from detection import SSDDetect
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt

VID_PATH = os.path.expanduser('~/clapton.mkv')
print 'Loading video...'
vidgen = skvideo.io.vreader(VID_PATH)

frames = []
i = 0
# Load one second of the video
for frame in vidgen:
    if i > 2489:
        frames.append(frame)
    i += 1
    if i == 2509:
        break

frames = np.array(frames)

WEIGHTS = 'data/frozen_aug.caffemodel'
DEPLOY = 'data/deploy.prototxt'

print 'Running detection...'
ssd = SSDDetect(DEPLOY, WEIGHTS)
frame_boxes = ssd.detect(frames)

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

for frame, boxes in zip(frames, frame_boxes):
    plt.imshow(frame)
    curr_axis = plt.gca()
    for coords, score, label, label_name in boxes:
        color = colors[label]
        curr_axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        display_txt = '%s: %.2f'%(label_name, score)
        curr_axis.text(coords[0][0], coords[0][1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
    plt.show()

plt.close()
