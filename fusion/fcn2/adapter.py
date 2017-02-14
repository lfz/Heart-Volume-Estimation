from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, GlobalPoolLayer, Upscale2DLayer
from lasagne.layers import ElemwiseSumLayer, NonlinearityLayer, SliceLayer, ConcatLayer, ScaleLayer
from lasagne.layers import dropout, batch_norm
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.init import GlorotNormal, GlorotUniform, HeUniform, HeNormal
from lasagne.objectives import squared_error, categorical_crossentropy, categorical_accuracy, binary_accuracy
import lasagne
import theano.tensor as T
import numpy as np
import random
import theano
import os
import pandas as pd
import cv2
import re

def my_softmax(x):
    e_x = T.exp(x-x.max(axis=1,keepdims=True))
    y = e_x / e_x.sum(axis=1, keepdims=True)
    return y

def build_fcn(input_var, inner_size):
    l_in = InputLayer(shape=(None, 1) + inner_size, input_var = input_var)

    # stage 1
    conv1_1 = batch_norm(Conv2DLayer(l_in, num_filters=32, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    conv1_2 = batch_norm(Conv2DLayer(conv1_1, num_filters=32, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    conv1_3 = batch_norm(Conv2DLayer(conv1_2, num_filters=32, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    pool1 = MaxPool2DLayer(conv1_3, pool_size=(2, 2))

    # stage 2
    conv2_1 = batch_norm(Conv2DLayer(pool1, num_filters=32, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    conv2_2 = batch_norm(Conv2DLayer(conv2_1, num_filters=32, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    conv2_3 = batch_norm(Conv2DLayer(conv2_2, num_filters=32, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    pool2 = MaxPool2DLayer(conv2_3, pool_size=(2, 2))
    
    # stage 3
    conv3_1 = batch_norm(Conv2DLayer(pool2, num_filters=64, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    conv3_2 = batch_norm(Conv2DLayer(conv3_1, num_filters=64, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    pool3 = MaxPool2DLayer(conv3_2, pool_size=(2, 2))
    
    # stage 3
    conv4_1 = batch_norm(Conv2DLayer(pool3, num_filters=64, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))
    conv4_2 = batch_norm(Conv2DLayer(conv4_1, num_filters=64, filter_size=(3, 3), 
                                     nonlinearity=rectify, W=HeNormal(),pad=1))

    
    # top-down stage 0
    l4_conv = batch_norm(Conv2DLayer(conv4_2, num_filters=5, filter_size=(1, 1),nonlinearity=rectify, W=HeNormal()))
    up4 = Upscale2DLayer(l4_conv, (8, 8))

    l3_conv = batch_norm(Conv2DLayer(conv3_2, num_filters=5, filter_size=(1, 1),nonlinearity=rectify, W=HeNormal()))
    up3 = Upscale2DLayer(l3_conv, (4, 4))

    l2_conv = batch_norm(Conv2DLayer(conv2_3, num_filters=5, filter_size=(1, 1),nonlinearity=rectify, W=HeNormal()))
    up2 = Upscale2DLayer(l2_conv, (2, 2))
    
    l1_conv = batch_norm(Conv2DLayer(conv1_3, num_filters=5, filter_size=(1, 1),nonlinearity=rectify, W=HeNormal()))

    concat = ConcatLayer([up4, up3, up2, l1_conv])
    
    mid1 = batch_norm(Conv2DLayer(concat, num_filters=20, filter_size=(3, 3),nonlinearity=rectify, W=HeNormal(),pad=1))
    mid2 = batch_norm(Conv2DLayer(mid1, num_filters=20, filter_size=(3, 3),nonlinearity=rectify, W=HeNormal(),pad=1))
    pred = Conv2DLayer(mid2, num_filters=2, filter_size=(3, 3),nonlinearity=my_softmax, W=HeNormal(),pad=1)
    concat2 = ConcatLayer([concat,mid1,mid2,pred])
    area1 = batch_norm(Conv2DLayer(concat2, num_filters=16, filter_size=(3, 3),nonlinearity=rectify, W=HeNormal(),pad=1))
    mid3 = Conv2DLayer(area1, num_filters=1, filter_size=(1, 1), W=HeNormal())
    area = GlobalPoolLayer(mid3)
    
    return pred, mid3, area
class adapter():
    def __init__(self, inner_size=None, snapshot_full_path=None):
        assert inner_size is not None
        assert snapshot_full_path is not None
        self.inner_size = inner_size
        input_var = T.tensor4('x')
        _, mid, area = build_fcn(input_var, self.inner_size)
        if os.path.exists(snapshot_full_path):
            with np.load(snapshot_full_path) as f:
                param_values = [f['arr_{}'.format(i)] for i in range(len(f.files))]
            print('resuming snapshot from {}'.format(snapshot_full_path))
            param_cur = lasagne.layers.get_all_params(area)
            assert len(param_cur) == len(param_values)
            for p, v in zip(param_cur, param_values):
                p.set_value(v)
        else:
            raise ValueError, "snapshot {} not found".format(snapshot_full_path)
        
        out = lasagne.layers.get_output(mid, deterministic=True)
        self.fn_forward = theano.function([input_var], out)        
        
    def convert(self, x_data):
        assert x_data.ndim == 4
        shape = x_data.shape
        outer_size = (shape[2], shape[3])
        
        # resize to inner size
        x_data_resized = np.zeros((shape[0], shape[1], self.inner_size[0], self.inner_size[1]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                x_data_resized[i, j, :, :] = cv2.resize(x_data[i, j, :, :], self.inner_size)
                
        # forward
        pred_data = self.fn_forward(x_data_resized.astype('float32'))
        
        # resize back
        shape = pred_data.shape
        pred_data_resized = np.zeros((shape[0], shape[1], outer_size[0], outer_size[1]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                pred_data_resized[i, j, :, :] = cv2.resize(pred_data[i, j, :, :], outer_size)
                
        # return
        return pred_data_resized.astype('float32')