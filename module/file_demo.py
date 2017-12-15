import os
from detection import SSDDetect

VID_NAME = 'bookofsouls.mp4'
VID_PATH = os.path.expanduser('~/Videos/' + VID_NAME)

WEIGHTS = 'data/frozen_aug.caffemodel'
DEPLOY = 'data/deploy_finetune.prototxt'
LABELMAP = 'data/ILSVRC2016/labelmap_ilsvrc_finetune.prototxt'

print 'Running detection...'
ssd = SSDDetect(DEPLOY, WEIGHTS, LABELMAP)
# You may have to lower this value to get good results
start = 0
conf_thresh = 0.3
boxgen = ssd.detect_vid(VID_PATH, start, 10000, conf_thresh)

frame_count = start
f = open(VID_NAME + '_result.txt', 'w')
for batch, frame_boxes in boxgen:
    for frame, boxes in zip(batch, frame_boxes):
        f.write(str(frame_count) + ' ')
        for box in boxes:
            labelnum = box[2]
            if labelnum != 1 and labelnum != 3 and labelnum != 4 and labelnum != 7:
                f.write(str(box) + ' ')
        frame_count += 1
        f.write('\n')

f.close()
