from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, GlobalPoolLayer, Upscale2DLayer
from lasagne.layers import ElemwiseSumLayer, NonlinearityLayer, SliceLayer, ConcatLayer, ScaleLayer
from lasagne.layers import dropout, batch_norm
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.init import GlorotNormal, GlorotUniform, HeUniform, HeNormal
from lasagne.objectives import squared_error, categorical_crossentropy, categorical_accuracy, binary_accuracy
import lasagne
import theano
import theano.tensor as T
import random
import pandas as pd
import re
import sys
import numpy as np
import os
import cv2


def read_csv(file_path):
    print("reading csv file from {}".format(file_path))
    volume_csv = pd.read_csv(file_path)
    volume_data = np.array(volume_csv.iloc[:, 1:3])
    rows = volume_data.shape[0]
    volume_data = volume_data.flatten().astype('float32')
    return volume_data

def preprocess(img):
    meanval = np.mean(img[img>0])     
    scaledIm = img.astype('float')/meanval/4*255
    scaledIm[scaledIm>255]=255
    scaledIm = scaledIm.astype('uint8')
    return scaledIm

def preprocess_patch(img):
    meanval = np.mean(img)
    std = np.std(img)
    return (img-meanval)/std


def preprocessSeries(imseries):
    meanval = np.mean(imseries[imseries>0])     
    scaledIm = (imseries.astype('float')-meanval)/meanval
    return scaledIm

def build_volume2(area, location, resolution, fixed_size):
    def _step(idx_, prior_result_, area_, location_, resolution_):
        area1 = area_[idx_] * T.prod(resolution_[idx_])
        area2 = area_[idx_ + 1] * T.prod(resolution_[idx_ + 1])
        h = location_[idx_ + 1] - location_[idx_]
        volume = (area1 + area2 )* np.prod(fixed_size).astype('float32') * h / 2.0 / 1000
        return prior_result_ + volume

    predict_V_list, _ = theano.scan(fn=_step,
                              outputs_info = np.array([0.]).astype('float32'),
                              sequences = T.arange(1000),
                              non_sequences = [area, location, resolution],
                              n_steps = location.shape[0] - 1)
    predict_V = predict_V_list[-1]
    return predict_V[0]

def build_volume3(area, location, resolution, fixed_size):
    def _step(idx_, prior_result_, area_, location_, resolution_):
        area1 = area_[idx_] * T.prod(resolution_[idx_])
        area2 = area_[idx_ + 1] * T.prod(resolution_[idx_ + 1])
        h = location_[idx_ + 1] - location_[idx_]
        volume = (area1 + area2 + T.sqrt(area1 * area2))* np.prod(fixed_size).astype('float32') * h / 3.0 / 1000
        return prior_result_ + volume

    predict_V_list, _ = theano.scan(fn=_step,
                              outputs_info = np.array([0.]).astype('float32'),
                              sequences = T.arange(1000),
                              non_sequences = [area, location, resolution],
                              n_steps = location.shape[0] - 1)
    predict_V = predict_V_list[-1]
    return predict_V[0]

def stage3_load_single_record(file_path, fixed_size):
    data = np.load(file_path).item()
    patch_list = data['patchStack']
    location_list = np.array(data['SliceLocation'])
    resolution = np.array(data['PixelSpacing'])
    resized_resolution_list = []
    resized_patch_list = []
    for patch in patch_list:
        resized_resolution_list.append(
            (resolution[0] / fixed_size[0] * patch.shape[0], resolution[1] / fixed_size[1] * patch.shape[1]))
        resized_patch_list.append(cv2.resize(patch, fixed_size))
    
    resized_patch_list = np.array(resized_patch_list, dtype='float32')[:, None, :, :]
    location_list = np.array(location_list, dtype='float32')
    resized_resolution_list = np.array(resized_resolution_list, dtype='float32')
    return resized_patch_list, location_list, resized_resolution_list



def aug(img,label, roi, angle_max = 1, scale_max = 1, offset_max=0, 
        flip_toggle=False, output_size=(50, 50), is_verbose=False):
    img = np.copy(img).astype(float)
    label = np.copy(label)
    x1, y1, x2, y2 = roi

    # adjust the roi
    assert x2 >= x1
    assert y2 >= y1
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    max_side = max(x2 - x1 + 1, y2 - y1 + 1)
    x1 = int(max(x_center - max_side / 2.0, 0))
    y1 = int(max(y_center - max_side / 2.0, 0))
    x2 = int(min(x_center + max_side / 2.0, img.shape[1]))
    y2 = int(min(y_center + max_side / 2.0, img.shape[0]))

    # do rotation
    angle = np.random.randint(int(angle_max))
    if is_verbose: 
        print("angle:", angle)
    rot_mat = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)
    rotated_mat = cv2.warpAffine(img, rot_mat, img.shape)
    rotated_mat2 = cv2.warpAffine(label, rot_mat, img.shape)

    # do scaling
    assert scale_max >= 1
    scaling = np.random.rand() * (scale_max - 1) + 1.2
    #scaling = scale_max
    if is_verbose:
        print("scaling: ", scaling)
    max_side = max_side * scaling
    x1 = int(max(x_center - max_side / 2.0, 0))
    y1 = int(max(y_center - max_side / 2.0, 0))
    x2 = int(min(x_center + max_side / 2.0, rotated_mat.shape[1]))
    y2 = int(min(y_center + max_side / 2.0, rotated_mat.shape[0]))

    # do offset
    x_offset = int((2 * np.random.rand() - 1 ) * offset_max*max_side*(scaling-1)/2)
    y_offset = int((2 * np.random.rand() - 1 ) * offset_max*max_side*(scaling-1)/2)
    if is_verbose:
        print("x_offset: ", x_offset, "y_offset: ", y_offset)
    x1 = min(max(x1 + x_offset, 0), rotated_mat.shape[1])
    y1 = min(max(y1 + y_offset, 0), rotated_mat.shape[0])
    x2 = min(max(x2 + x_offset, 0), rotated_mat.shape[1])
    y2 = min(max(y2 + y_offset, 0), rotated_mat.shape[0])
    patch = rotated_mat[y1:y2, x1:x2]
    new_area = patch.shape[0] * patch.shape[1]
    label_patch = rotated_mat2[y1:y2, x1:x2]

    # do flip if on
    if flip_toggle:
        if np.random.rand() > 0.5:
            if is_verbose:
                print("is_flip: True")
            patch = patch[::-1, ::-1]
            label_patch =  label_patch[::-1, ::-1]
        elif is_verbose:
            print("is_flip: False")

    # resize
    patch = cv2.resize(patch, output_size)
    label_patch = cv2.resize(label_patch, output_size)
    return patch.astype('float32'), new_area,label_patch

def my_aug(patch, label_patch, fixed_size):
    patch = cv2.resize(patch, fixed_size)
    label_patch = cv2.resize(label_patch, fixed_size)
    patches = []
    label_patches = []
    for i in range(4):
        patches.append(patch)
        patches.append(patch[::-1, :])
        patches.append(patch[:, ::-1])
        patches.append(patch[::-1, ::-1])
        label_patches.append(label_patch)
        label_patches.append(label_patch[::-1, :])
        label_patches.append(label_patch[:, ::-1])
        label_patches.append(label_patch[::-1, ::-1])
        patch = cv2.transpose(patch)
        label_patch = cv2.transpose(label_patch)
    return patches, label_patches

def extendBox(box,upper,ratio):
    margin1 = np.min([ratio/2*box[2]+8,box[0],box[1]])
    margin2 = np.min([upper[0]-box[2],upper[1]-box[3],box[2]*ratio+16])
    tmp = np.array([box[0]-margin1,box[1]-margin1,box[2]+margin2,box[3]+margin2]).astype('int')
    return tmp

def augPos(data):
    rect = data['square']
    label = data['label']
    mid = np.mean(rect,axis=0).astype('int')
    length =np.max(rect[1]-rect[0])/2
    box = np.concatenate([mid-length,[length*2,length*2]])
    img = data['img']
    patchbag = []
    labelbag = []
    shape = img.shape
    for ratio in np.linspace(0.25,2,4):
        box2 = extendBox(box,shape,ratio)
        patchbag.append(preprocess_patch(img[box2[1]:box2[1]+box2[3],box2[0]:box2[0]+box2[2]]))
        labelbag.append(label[box2[1]:box2[1]+box2[3],box2[0]:box2[0]+box2[2]])
    return patchbag,labelbag

def load_pos_samples(root_dir, fixed_size):
    all_patch = []
    all_label = []
    augratio = 30
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy'):
                full_path = os.path.join(root, file)
                data = np.load(full_path).item()
                label_patch = data['labelpatch'].astype('float32')
                patch = data['patch']
                img = data['img']
                label = data['label']
                roi = data['square'].reshape([-1])
                assert label_patch.shape[0] == patch.shape[0]
                assert label_patch.shape[1] == patch.shape[1]
                for i in range(augratio):
                    patch, new_area,label_patch = aug(img,label, roi, angle_max = 1, scale_max =3, offset_max=0.3, 
        flip_toggle=True, output_size=fixed_size, is_verbose=False)
                    all_patch.append(preprocess_patch(patch))
                    all_label.append(label_patch)
    all_patch = np.array(all_patch, dtype='float32')
    all_label = np.array(all_label, dtype='float32')
    print("loaded pos_patch: {}, pos_label: {} from {}".format(all_patch.shape, all_label.shape, root_dir))
    return all_patch, all_label


def load_neg_samples2(file, fixed_size):
    all_patch = []
    all_label = []
    imgs = np.load(file)
    for patch in imgs:
        label_patch = np.zeros(patch.shape).astype('float32')
        patches, label_patches = my_aug(preprocess_patch(patch), label_patch, fixed_size)
        all_patch.extend(patches)
        all_label.extend(label_patches)
    all_patch = np.array(all_patch, dtype='float32')
    all_label = np.array(all_label, dtype='float32')
    print("loaded neg_patch: {}, neg_label: {} from {}".format(all_patch.shape, all_label.shape, file))
    return all_patch, all_label

def getData(root_dir, fixed_size):
    train_patch_pos, train_label_pos = load_pos_samples(os.path.join(root_dir, 'train'), fixed_size)
    val_patch_pos, val_label_pos = load_pos_samples(os.path.join(root_dir, 'val'), fixed_size)
    train_patch_pos = np.concatenate([train_patch_pos,val_patch_pos],axis=0)
    train_label_pos = np.concatenate([train_label_pos,val_label_pos],axis=0)
    print(val_patch_pos.shape)
    val_patch_pos = val_patch_pos[:1,:,:]
    val_label_pos = val_label_pos[:1,:,:]

    # patch_neg, label_neg = load_neg_samples('heart_area_data/data_stage2/neg/', fixed_size)
    patch_neg, label_neg = load_neg_samples2(os.path.join(root_dir, 'neg.npy'), fixed_size)
    
    ratio = 0.0
    train_num_pos = len(train_patch_pos)
    val_num_pos = len(val_patch_pos)

    # desired negative sampels number
    train_num_neg = int(train_num_pos * ratio)
    val_num_neg = int(val_num_pos * ratio)

    idx_train_neg = np.random.randint(len(patch_neg), size=train_num_neg)
    idx_val_neg = np.random.randint(len(patch_neg), size=val_num_neg)
    train_patch_neg = patch_neg[idx_train_neg]
    train_label_neg = label_neg[idx_train_neg]
    val_patch_neg = patch_neg[idx_val_neg]
    val_label_neg = label_neg[idx_val_neg]

    train_patch = np.concatenate([train_patch_pos, train_patch_neg], axis=0)
    train_label = np.concatenate([train_label_pos, train_label_neg], axis=0)
    val_patch = np.concatenate([val_patch_pos, val_patch_neg], axis=0)
    val_label = np.concatenate([val_label_pos, val_label_neg], axis=0)
    print("ratio: {}".format(ratio))
    print("train_patch: pos {}, neg {}, total {}".format(train_patch_pos.shape, train_patch_neg.shape, train_patch.shape))
    print("train_label: pos {}, neg {}, total {}".format(train_label_pos.shape, train_label_neg.shape, train_label.shape))
    print("val_patch: pos {}, neg {}, total {}".format(val_patch_pos.shape, val_patch_neg.shape, val_patch.shape))
    print("val_label: pos {}, neg {}, total {}".format(val_label_pos.shape, val_label_neg.shape, val_label.shape))
    
    return train_patch,train_label,val_patch,val_label


def load_patch_stacks(root_dir, fixed_size):
    print("reading all patch stacks from {}".format(root_dir))
    x_data = []
    location_data = []
    resolution_data = []
    counter = 0
    for patient_id in sorted(os.listdir(root_dir), key=int):
        counter += 1
        full_root_path = os.path.join(root_dir, patient_id)
        x_data_batch = []
        location_data_batch = []
        resolution_data_batch = []
        for file in sorted(os.listdir(full_root_path), key=lambda x : int(x[:-4])):
            file_full_path = os.path.join(full_root_path, file)
            x_data_e, location_data_e, resolution_data_e = stage3_load_single_record(file_full_path, fixed_size)
            x_data_batch.append(x_data_e)
            location_data_batch.append(location_data_e)
            resolution_data_batch.append(resolution_data_e)
        x_data.append(x_data_batch)
        location_data.append(location_data_batch)
        resolution_data.append(resolution_data_batch)
        if counter % 50 == 0:
            print(patient_id)
    return x_data, location_data, resolution_data

def load_min_max_patch_stacks(root_dir, fixed_size):
    min_root_dir = os.path.join(root_dir, 'min')
    max_root_dir = os.path.join(root_dir, 'max')
    x_data = []
    location_data = []
    resolution_data = []
    rows = len(os.listdir(min_root_dir))
    for i in range(1, rows + 1):
        min_full_path = os.path.join(min_root_dir, str(i) + '.npy')
        max_full_path = os.path.join(max_root_dir, str(i) + '.npy')
        paths = [min_full_path, max_full_path]
        for path in paths:
            x_data_single, location_data_single, resolution_data_single = stage3_load_single_record(path, fixed_size)
            x_data.append(x_data_single)
            location_data.append(location_data_single)
            resolution_data.append(resolution_data_single)
    assert len(x_data) == len(location_data)
    assert len(location_data) == len(resolution_data)
    return x_data, location_data, resolution_data