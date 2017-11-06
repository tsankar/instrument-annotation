'''
Data assumed to be in MXNet .rec file format
Data can be formatted as such using im2rec.py as released by MXNet

Modeled off of code from https://mxnet.incubator.apache.org/how_to/finetune.html
'''

import mxnet as mx
import os
import sys
os.chdir('/home/tharun/Deformable-ConvNets/rfcn')
print os.getcwd()

sys.path.append(os.path.dirname(os.path.expanduser('~/Deformable-ConvNets/rfcn/symbols')))
sys.path.append(os.path.dirname(os.path.expanduser('~/Deformable-ConvNets/lib/utils')))
sys.path.append(os.path.dirname(os.path.expanduser('~/Deformable-ConvNets/rfcn')))
from rfcn.symbols import *
from rfcn.config.config import config, update_config
import cv2
import numpy as np
import pprint

update_config('transfer_config.yaml')

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = '''TODO''',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = '''TODO''',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)

pprint.pprint(config)
symbol = 'resnet_v1_101_rfcn_dcn'
sym_instance = eval(symbol + '.' + symbol)()
sym = sym_instance.get_symbol(config, is_train=True)

def get_fine_tune_model(symbol, arg_params, num_classes, layer_names=['cls_prob_reshape', 'rpn_cls_prob']):
    all_layers = symbol.get_internals()
    net1 = all_layers[layer_names[0]+'_output']
    net2 = all_layers[layer_names[1]+'_output']
    # net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net1 = mx.symbol.SoftmaxOutput(data=net1, name='softmax_cls')
    net2 = mx.symbol.SoftmaxOutput(data=net2, name='softmax_rpn')
    new_args = dict({k:arg_params[k] for k in arg_params if 'softmax_cls' not in k and 'softmax_rpn' not in k})
    group = mx.sym.Group([net1, net2])
    return (group, new_args)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
        num_epoch=8,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

arg_params, aux_params = load_param(cur_path + '/../model/' + ('rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'), 0, process=True)
(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
batch_size = 32
num_gpus = 1
(train, val) = get_iterators(batch_size)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
print 'Training accuracy: ' + mod_score

mod.save('instruments', 8)
