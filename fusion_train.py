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

def fusion_train(x_data_train, location_data_train, resolution_data_train, volume_data_train,
                 x_data_val, location_data_val, resolution_data_val, volume_data_val,
                 fixed_size, n_epoches = 2, lr = 0.001):
    loss_epoches = []
    snapshot_root = 'fusion_snapshot'
    n_train_samples = len(x_data_train)
    n_val_samples = len(x_data_val)


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

    # area, 1d vector
    train_area = lasagne.layers.get_output(l_out, deterministic=True).flatten()
    val_area = lasagne.layers.get_output(l_out, deterministic=True).flatten()

    # predict volume, 0d scalar
    train_pred_volume = utee.build_volume2(train_area, location, resolution, fixed_size) 
    val_pred_volume = utee.build_volume2(val_area, location, resolution, fixed_size)

    # loss, 0d scalar
    train_loss = T.abs_(train_pred_volume - target_volume).mean() / 600.
    val_loss = T.abs_(val_pred_volume - target_volume).mean() / 600.

    # params
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    
    fusion_snapshot_path ='fusion_snapshot/0.npz'
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
        
    #params[0].set_value(np.ones((1,6,1,1),dtype='float32')/6.)
    #print("snapshot to {}".format("0.npz"))
    #np.savez("0.npz", *lasagne.layers.get_all_param_values(l_out))
                 
    print(params[0].get_value(), params[0].get_value().shape)
    updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=lr, momentum=0.9)

    train_fn = theano.function(
        [pred, location, resolution, target_volume],
        train_loss,
        updates = updates
    )
    val_fn = theano.function(
        [pred, location, resolution, target_volume],
        val_loss
    )
    test_fn = theano.function(
        [pred, location, resolution],
        [val_area, val_pred_volume]
    )

    area_fn = theano.function(
        [pred],
        val_area
    )


    print("Training and validating precedure begin")
    for cur_epoch in range(n_epoches+1):
        print("epoch {}/{} begin".format(cur_epoch, n_epoches))

        if cur_epoch > 0:
            print(".training, {} samples to go".format(n_train_samples))
            losses_data_train = []
            for j in range(n_train_samples):
                x_e = x_data_train[j].astype('float32')
                preds = []
                for adapter in adapters:
                    preds.append(adapter.convert(x_e))
                pred_e = np.concatenate(preds, axis=1)
                location_e = location_data_train[j].astype('float32')
                resolution_e = resolution_data_train[j].astype('float32')
                volume_e = volume_data_train[j].astype('float32')
                loss_data_train = train_fn(pred_e, location_e, resolution_e, volume_e)
                losses_data_train.append(loss_data_train)
            print(".training loss: {}".format(np.mean(loss_data_train)))
            if np.isnan(np.mean(loss_data_train)):
                print(".training detect nan, break and stop")
                break
                
        print(".validating, {} samples to go".format(n_val_samples))
        losses_data_val = []
        for i in range(n_val_samples):
            volume_min_e = volume_data_val[2*i].astype('float32')
            volume_max_e = volume_data_val[2*i+1].astype('float32')
            pred_volumes_data = []
            for j in range(len(x_data_val[i])):
                x_e = x_data_val[i][j].astype('float32')
                preds = []
                for adapter in adapters:
                    preds.append(adapter.convert(x_e))
                pred_e = np.concatenate(preds, axis=1)
                location_e = location_data_val[i][j].astype('float32')
                resolution_e = resolution_data_val[i][j].astype('float32')
                _, pred_volume_data = test_fn(pred_e, location_e, resolution_e)
                if np.isnan(pred_volume_data):
                    print(x_e, pred_e, location_e, resolution_e)
                pred_volumes_data.append(pred_volume_data)
            volume_min_pred = np.min(pred_volumes_data)
            volume_max_pred = np.max(pred_volumes_data)
            loss_min_data = np.abs(volume_min_pred - volume_min_e) / 600
            loss_max_data = np.abs(volume_max_pred - volume_max_e) / 600
            loss_data_val = (loss_min_data + loss_max_data) / 2.0
            losses_data_val.append(loss_data_val)
        print(".validating loss: {}".format(np.mean(losses_data_val)))
        loss_epoches.append(np.mean(losses_data_val))
        if np.isnan(np.mean(loss_data_val)):
            print(".training detect nan, break and stop")
            break

        cur_snapshot_path = os.path.join(snapshot_root, str(cur_epoch) + '.npz')
        print("snapshot to {}".format(cur_snapshot_path))
        np.savez(cur_snapshot_path, *lasagne.layers.get_all_param_values(l_out))
    print("Training done!")


    idx = np.argmin(loss_epoches)
    with open('./SETTINGS.json', 'r') as f:    
        data = json.load(f)
    best_snapshot_path = os.path.join(snapshot_root, str(idx) + '.npz')
    print("add info {} to SETTINGS.json".format(best_snapshot_path))
    data['FUSION_SNAPSHOT_PATH'] = os.path.join(best_snapshot_path)
    with open('./SETTINGS.json', 'w') as f:
        json.dump(data, f)