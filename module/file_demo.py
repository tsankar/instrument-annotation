import os
from detection import SSDDetect

VID_NAME = 'slash.mp4'
VID_PATH = os.path.expanduser('~/Videos/' + VID_NAME)

WEIGHTS = 'data/frozen_aug.caffemodel'
DEPLOY = 'data/deploy_finetune.prototxt'
LABELMAP = 'data/ILSVRC2016/labelmap_ilsvrc_finetune.prototxt'

print 'Running detection...'
ssd = SSDDetect(DEPLOY, WEIGHTS, LABELMAP)
# You may have to lower this value to get good results
conf_thresh = 0.3
boxgen = ssd.detect_vid(VID_PATH, 0, 455, conf_thresh)

frame_count = 0
f = open(VID_NAME + '_result.txt', 'w')
for batch, frame_boxes in boxgen:
    for frame, boxes in zip(batch, frame_boxes):
        f.write(str(frame_count) + ' ')
        for box in boxes:
            f.write(str(box) + ' ')
        frame_count += 1
        f.write('\n')

f.close()
