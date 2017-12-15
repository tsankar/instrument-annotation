import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import os
import caffe
import hashlib
import skvideo.io
import skimage.io

def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    From keras/data_utils.py
    # Arguments
    fpath: path to the file being validated
    file_hash:  The expected hash string of the file.
    The sha256 and md5 hash algorithms are both supported.
    algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
    The default 'auto' detects the hash algorithm in use.
    chunk_size: Bytes to read at a time, important for large files.
    # Returns
    Whether the file is valid
    """
    if ((algorithm is 'sha256') or (algorithm is 'auto' and len(file_hash) is 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    From keras/data_utils.py
    # Example
    ```python
    >>> from keras.data_utils import _hash_file
    >>> _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    # Arguments
    fpath: path to the file being validated
    algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
    The default 'auto' detects the hash algorithm in use.
    chunk_size: Bytes to read at a time, important for large files.
    # Returns
    The file hash
    """
    if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_labelname(labelmap, labels):
    """
    Retrieves human-readable labels from labelmap file
    From ssd_detect.ipynb in SSD repo
    """
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class SSDDetect:
    def __init__(self, def_path, weights_path,
    labelmap_path=os.path.expanduser('~/SSD-instruments/data/ILSVRC2016/labelmap_ilsvrc_det.prototxt')):
        """
        weights_path = path to model weights (.caffemodel)
        def_path = path to model definition (deploy.prototxt)
        """
        f = open(labelmap_path, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(f.read()), self.labelmap)

        # expected_hash = ''
        #TODO: versioning
        # if not validate_file(weights_path, expected_hash):
        #     # TODO: Download correct weights
        #     weights_path = ''

        self.net = caffe.Net(def_path, weights_path, caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        caffe.set_device(0)
        caffe.set_mode_gpu()


    # TODO: adjust for image batches? - determine whether necessary
    def detect(self, frames, conf_thresh, batch_size=1):
        '''
        frames: numpy array (of RGB images) of shape
            (number of frames, height, width, channels)
        Returns list of list of boxes for each frame. Each box is
        formatted as a tuple ((xmin, ymin, w, h), score, label)
        '''
        # image = caffe.io.load_image(image_path)

        image_resize = 300
        self.net.blobs['data'].reshape(batch_size, 3, image_resize, image_resize)

        frame_boxes = []

        for image in frames:
            transformed_image = self.transformer.preprocess('data', image)
            self.net.blobs['data'].data[...] = transformed_image

            # Forward pass.
            detections = self.net.forward()['detection_out']

            # Parse the outputs.
            det_label = detections[0,0,:,1]
            det_conf = detections[0,0,:,2]
            det_xmin = detections[0,0,:,3]
            det_ymin = detections[0,0,:,4]
            det_xmax = detections[0,0,:,5]
            det_ymax = detections[0,0,:,6]

            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_labels = get_labelname(self.labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            boxes = []
            # Tuples formatted (((xmin, ymin), w, h), score, label, labelname)
            # where labelname is human-readable label
            for i in xrange(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(round(top_ymax[i] * image.shape[0]))
                score = top_conf[i]
                label = int(top_label_indices[i])
                label_name = top_labels[i]
                coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
                tup = (coords, score, label, label_name)
                boxes.append(tup)

            new_boxes = []
            skips = []
            # Remove overlaps
            if len(boxes) > 1:
                i = 0
                while i < len(boxes):
                    if i not in skips:
                        j = i + 1
                        while j < len(boxes):
                            if j not in skips:
                                boxA = boxes[i][0]
                                xa1 = boxA[0][0]
                                ya1 = boxA[0][1]
                                A = {'x1': xa1, 'y1': ya1, 'x2': xa1 + boxA[1] - 1,
                                'y2': ya1 + boxA[2] - 1}

                                boxB = boxes[j][0]
                                xb1 = boxB[0][0]
                                yb1 = boxB[0][1]
                                B = {'x1': xb1, 'y1': yb1, 'x2': xb1 + boxB[1] - 1,
                                'y2': yb1 + boxB[2] - 1}

                                SI = np.max([ 0, 1 + np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * \
                                np.max([ 0, 1 + np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
                                s = boxA[1] * boxA[2] + boxB[1] * boxB[2] - SI
                                ratio = float(SI) / float(s)

                                # print 'Ratio', ratio

                                if ratio > 0.8:
                                    if boxes[i][1] > boxes[j][1]:
                                        skips.append(j)
                                    else:
                                        skips.append(i)
                                        break
                            j += 1
                    i += 1

                for k in xrange(len(boxes)):
                    if k not in skips:
                        new_boxes.append(boxes[k])


            else:
                new_boxes = boxes

            frame_boxes.append(new_boxes)

        return frame_boxes


    def detect_vid(self, vid_path, start, length, conf_thresh):
        vidgen = skvideo.io.vreader(vid_path)
        num_frames = int(skvideo.io.ffprobe(vid_path)['video']['@nb_frames'])

        frame_start = start
        print 'Frame start', frame_start
        frame_length = length * 30

        i = 0
        j = 0
        batch = []
        for frame in vidgen:
            if j >= frame_start:
                frame_float = skimage.img_as_float(frame).astype(np.float32)
                batch.append(frame_float)
                if i == 29 or num_frames - j == 1:
                    i = 0
                    boxes = self.detect(np.array(batch), conf_thresh)
                    yield batch, boxes
                    batch = []
                else:
                    i += 1
                if j == frame_start + frame_length:
                    break
            j += 1
