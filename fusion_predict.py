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
import sys
import json
import benchmark as bm
import utee
from fusion.fcn1.adapter import adapter as adapter1
from fusion.fcn2.adapter import adapter as adapter2
from fusion.fcn3.adapter import adapter as adapter3
from fusion.fcn4.adapter import adapter as adapter4
from fusion.fcn5.adapter import adapter as adapter5
from fusion.fcn6.adapter import adapter as adapter6

def fusion_predict(fusion_snapshot_path, out_test_data_path, stage2_data_root_dir, submist_save_file_path, fixed_size):
    # all adapters
    adapters = []
    adapters.append(adapter1((48, 48), 'fusion/fcn1/96.npz'))
    adapters.append(adapter2((48, 48), 'fusion/fcn2/92.npz'))
    adapters.append(adapter3((48, 48), 'fusion/fcn3/52.npz'))
    adapters.append(adapter4((48, 48), 'fusion/fcn4/470.npz'))
    adapters.append(adapter5((48, 48), 'fusion/fcn5/280.npz'))
    adapters.append(adapter6((48, 48), 'fusion/fcn6/114.npz'))

    # input tensor
    pred = T.tensor4('pred')
    location = T.vector('location')
    resolution = T.matrix('resolution')
    target_volume = T.fscalar('volume')


    # fusion layers
    l_in = InputLayer(shape=(None, len(adapters), fixed_size[0], fixed_size[1]), input_var = pred)
    mid = Conv2DLayer(l_in, num_filters=1, filter_size=(1, 1), W=HeNormal())
    l_out = GlobalPoolLayer(mid)


    test_area = lasagne.layers.get_output(l_out, deterministic=True).flatten()

    test_pred_volume = utee.build_volume2(test_area, location, resolution, fixed_size)

    test_fn = theano.function(
        [pred, location, resolution],
        [test_area, test_pred_volume]
    )

    area_fn = theano.function(
        [pred],
        test_area
    )

    if os.path.exists(fusion_snapshot_path):
        with np.load(fusion_snapshot_path) as f:
            param_values = [f['arr_{}'.format(i)] for i in range(len(f.files))]
        print('resuming snapshot from {}'.format(fusion_snapshot_path))
        param_cur = lasagne.layers.get_all_params(l_out)
        assert len(param_cur) == len(param_values)
        for p, v in zip(param_cur, param_values):
            p.set_value(v)
    else:
        print("snapshot {} not found".format(fusion_snapshot_path))


    bm.fusion_submit(stage2_data_root_dir, out_test_data_path,
                      adapters, area_fn, test_fn, 
                      fixed_size, submist_save_file_path)