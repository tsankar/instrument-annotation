'''
Sample object detection script
To be modified to be more applicable to the task at hand

Code modeled off of demo.py from Deformable Convolutional Networks
'''

import os
import sys
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np

import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

# Setup
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

DCN_ROOT = '~/Deformable-ConvNets'
update_config(DCN_ROOT + '/experiments/rfcn/cfgs/rfcn_coco_demo.yaml')

def main():
    config.symbol = 'resnet_v1_101_rfcn_dcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
