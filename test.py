import lmdb
import caffe
import cv2
from caffe.proto import caffe_pb2
import numpy as np

lmdb_file = "/home/tharun/data/ILSVRC/lmdb/DET/ILSVRC2016_trainval1_lmdb_aug"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.AnnotatedDatum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    data = datum.datum
    grp = datum.annotation_group

    arr = np.frombuffer(data.data, dtype='uint8')
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    width = img.shape[1]
    height = img.shape[0]
    for annotation in grp:
        for bbox in annotation.annotation:
            cv2.rectangle(img, (int(bbox.bbox.xmin * width), int(bbox.bbox.ymin*height)), (int(bbox.bbox.xmax*width), int(bbox.bbox.ymax*height)), (0,255,0))
    cv2.imshow('decoded image', img)
    cv2.waitKey(3000)
    break
