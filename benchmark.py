import numpy as np
import random
import theano
import theano.tensor as T
import lasagne
import pandas as pd
from scipy.stats import norm
import os
import cv2
import utee
from utee import getData
from utee import load_patch_stacks, stage3_load_single_record

def volumeVar(Area,loc,Reso,shape,varianceTable):
    spacing = np.linspace(0,1,len(varianceTable))
    sumvar =0 
    for i in range(len(loc)):
        area = Area[i]
        reso = Reso[i]
        if i==0:
            alpha = (loc[1]-loc[0])*np.prod(reso)
        elif i==len(loc)-1:
            alpha = (loc[i] - loc[i-1])*np.prod(reso)
        else:
            alpha = (loc[i+1]-loc[i-1])*np.prod(reso)
        idx = np.sum(area>spacing)
        var = varianceTable[idx]
        sumvar = sumvar+alpha**2*var
    finalvar = sumvar*(1./2000*np.prod(shape))**2
    return finalvar

def varianceLUT(X,Y):
    z1 = np.polyfit(X, Y, 1) 
    p1 = z1[0]*X+z1[1]
    residual = p1-Y
    X,residual,Y =[np.array(k) for k in  zip(*sorted(zip(X,residual,Y),key=lambda x:x[0]))]
    finerX = np.linspace(0,1,1000)
    averageR=np.zeros_like(finerX)
    countX = np.zeros_like(finerX)
    for x,r in zip(X,residual):
        idx = sum(x>finerX)
        averageR[idx] = averageR[idx]+r**2
        countX[idx]=countX[idx]+1
    averageR[countX!=0] = averageR[countX!=0]/countX[countX!=0]
    smoothR = smooth(averageR,100)
    varianceTable = smoothR
    return varianceTable

def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),x,mode='same')
    return y

def convert_to_csv(diastole_mean, systole_mean, diastole_std=None, systole_std=None, 
                  submit_csv_path='submit.csv',
                  min_sig = 0.01):
    
    base_idx = 701
    rows = len(diastole_mean)
    columns = ['Id'] + ['P' + str(i) for i in range(600)]
    index = np.arange(2 * rows)
    csv_data = pd.DataFrame(index=index, columns=columns)
    
    assert len(diastole_mean) == len(systole_mean)
    if diastole_std is not None:
        assert len(diastole_std) == rows
    else:
        diastole_std = np.ones(rows) * min_sig
    if systole_std is not None:
        assert len(systole_std) == rows
    else:
        systole_std = np.ones(rows) * min_sig
        
    for i in range(rows):
        xx = np.linspace(0, 599, 600)
        
        # for diastole
        diastole_mean_e = diastole_mean[i]
        diastole_std_e = np.max(diastole_std[i], min_sig)
        yy = norm.cdf((xx - diastole_mean_e) / diastole_std_e)
        yy = yy / np.max(yy)
        csv_data.iloc[2*i, 0] = str(base_idx + i) + '_Diastole'
        csv_data.iloc[2*i , 1:] = yy
        
        # for systole
        systole_mean_e = systole_mean[i]
        systole_std_e = np.max(systole_std[i], min_sig)
        yy = norm.cdf((xx - systole_mean_e) / systole_std_e)
        yy = yy / np.max(yy)
        csv_data.iloc[2*i+1, 0] = str(base_idx + i) + '_Systole'
        csv_data.iloc[2*i+1, 1:] = yy
    
    print("saving to {}".format(submit_csv_path))
    csv_data.to_csv(submit_csv_path, index=False)

def test_on_train_data(stage3_test_fn, fixed_size):
    # read train and val volume data
    volume_csv_path = os.path.expanduser('~/data/kaggle/clean/stage3/train.csv')
    volume_csv = pd.read_csv(volume_csv_path)
    volume_data = np.array(volume_csv.iloc[:, 1:3])
    rows = volume_data.shape[0]
    volume_data = volume_data.flatten().astype('float32')

    # read train and val patch, location and resolution
    root_dir = os.path.expanduser('~/data/kaggle/clean/stage3')
    min_root_dir = os.path.join(root_dir, 'min')
    max_root_dir = os.path.join(root_dir, 'max')
    x_data = []
    location_data = []
    resolution_data = []
    for i in range(1, rows + 1):
        min_full_path = os.path.join(min_root_dir, str(i) + '.npy')
        max_full_path = os.path.join(max_root_dir, str(i) + '.npy')
        paths = [min_full_path, max_full_path]
        for path in paths:
            x_data_single, location_data_single, resolution_data_single = stage3_load_single_record(path, fixed_size)
            x_data.append(x_data_single)
            location_data.append(location_data_single)
            resolution_data.append(resolution_data_single)
    assert len(volume_data) == len(x_data)
    assert len(x_data) == len(location_data)
    assert len(location_data) == len(resolution_data)

    for i in range(rows):
        pred_volume = []
        for j in range(2):
            x_e = x_data[2*i+j].astype('float32')
            location_e = location_data[2*i+j].astype('float32')
            resolution_e = resolution_data[2*i+j].astype('float32')
            volume_e = volume_data[2*i+j].astype('float32')
            _, volume = stage3_test_fn(x_e, location_e, resolution_e)
            pred_volume.append(volume)
        print("id: {}, min: {} -> {}, max: {} -> {}".format(i+1, volume_data[2*i], pred_volume[0], 
                                                            volume_data[2*i+1], pred_volume[1]))

def fusion_test(adapters, fusion_test_fn, fixed_size):
    # FUSION read test patch, location and resolution
    x_data_test = []
    location_data_test = []
    resolution_data_test = []
    test_root_dir = '../clean/trainPatchStack/'
    for patient_id in sorted(os.listdir(test_root_dir), key=int):
        full_root_path = os.path.join(test_root_dir, patient_id)
        x_data_batch = []
        location_data_batch = []
        resolution_data_batch = []
        for file in sorted(os.listdir(full_root_path), key=lambda x : int(x[:-4])):
            file_full_path = os.path.join(full_root_path, file)
            x_data_e, location_data_e, resolution_data_e = stage3_load_single_record(file_full_path, fixed_size)
            x_data_batch.append(x_data_e)
            location_data_batch.append(location_data_e)
            resolution_data_batch.append(resolution_data_e)
        x_data_test.append(x_data_batch)
        location_data_test.append(location_data_batch)
        resolution_data_test.append(resolution_data_batch)
        if int(patient_id) % 50 == 0:
            print(patient_id)

    # FUSION read gt volume
    volume_csv_path = os.path.expanduser('~/data/kaggle/clean/stage3/train.csv')
    volume_csv = pd.read_csv(volume_csv_path)
    volume_data = np.array(volume_csv.iloc[:, 1:3])
    rows = volume_data.shape[0]
    gt_volumes = volume_data.flatten().astype('float32')

    print("TEST, x: {}, location: {}, resolution: {}, volume: {}".format(len(x_data_test), len(location_data_test), 
                                                             len(resolution_data_test), len(volume_data)))
    # FUSION make prediction and compute volume variance
    pred_volumes_diastole = []
    pred_volumes_systole = []
    fusion_n_test_batches = len(x_data_test)
    for i in range(fusion_n_test_batches):
        if i % 50 == 0:
            print(i)
        n_samples = len(x_data_test[i])
        volumes = []
        for j in range(n_samples):
            x_e = x_data_test[i][j].astype('float32')
            preds = []
            for adapter in adapters:
                preds.append(adapter.convert(x_e))
            pred_e = np.concatenate(preds, axis=1)
            location_e = location_data_test[i][j]
            resolution_e = resolution_data_test[i][j]
            _, volume = fusion_test_fn(pred_e, location_e, resolution_e)
            volumes.append(volume)

        volumes = np.array(volumes).flatten()
        pred_volumes_diastole.append(np.max(volumes))
        pred_volumes_systole.append(np.min(volumes))

    pred_volumes = np.array(zip(pred_volumes_systole, pred_volumes_diastole)).flatten()
    pred_volumes_diastole = np.array(pred_volumes_diastole)
    pred_volumes_systole = np.array(pred_volumes_systole)
    gt_volumes_systole = np.array([gt_volumes[i] for i in range(len(gt_volumes)) if i % 2 == 0])
    gt_volumes_diastole = np.array([gt_volumes[i] for i in range(len(gt_volumes)) if i % 2 == 1])
    print(len(pred_volumes), len(gt_volumes))
    print(len(pred_volumes_diastole), len(gt_volumes_diastole))
    print(len(pred_volumes_systole), len(gt_volumes_diastole))
    loss_total = np.abs(pred_volumes - gt_volumes[:len(pred_volumes)]).mean() / 600
    loss_systole = np.abs(pred_volumes_systole - gt_volumes_systole[:len(pred_volumes_systole)]).mean() / 600
    loss_diastole = np.abs(pred_volumes_diastole - gt_volumes_diastole[:len(pred_volumes_diastole)]).mean() / 600
    print("loss_total: {}, loss_systole: {}, loss_diastole: {}".format(loss_total, loss_systole, loss_diastole))
    
def sigle_submit(pred_areas, gt_areas, stage3_test_fn, fixed_size, file_path='submit.csv'):
    assert len(pred_areas) == len(gt_areas)
    var_lookup_table = varianceLUT(pred_areas, gt_areas)

     # STAGE3 read test patch, location and resolution
    x_data_test = []
    location_data_test = []
    resolution_data_test = []
    test_root_dir = '../clean/validatePatchStack/'
    for patient_id in sorted(os.listdir(test_root_dir), key=int):
        full_root_path = os.path.join(test_root_dir, patient_id)
        x_data_batch = []
        location_data_batch = []
        resolution_data_batch = []
        for file in sorted(os.listdir(full_root_path), key=lambda x : int(x[:-4])):
            file_full_path = os.path.join(full_root_path, file)
            x_data_e, location_data_e, resolution_data_e = stage3_load_single_record(file_full_path, fixed_size)
            x_data_batch.append(x_data_e)
            location_data_batch.append(location_data_e)
            resolution_data_batch.append(resolution_data_e)
        x_data_test.append(x_data_batch)
        location_data_test.append(location_data_batch)
        resolution_data_test.append(resolution_data_batch)        
    print("TEST, x: {}, location: {}, resolution: {}".format(len(x_data_test), len(location_data_test), 
                                                             len(resolution_data_test)))
    # STAGE3 make prediction and compute volume variance
    systole_mean = []
    diastole_mean = []
    systole_var = []
    diastole_var = []
    stage3_n_test_batches = len(x_data_test)
    for i in range(stage3_n_test_batches):
        n_samples = len(x_data_test[i])
        volumes = []
        volume_variances = []
        for j in range(n_samples):
            x_e = x_data_test[i][j]
            location_e = location_data_test[i][j]
            resolution_e = resolution_data_test[i][j]
            area, volume = stage3_test_fn(x_e, location_e, resolution_e)
            volume_variance = volumeVar(area.flatten(), location_e, resolution_e, fixed_size, var_lookup_table)
            volumes.append(volume[0])
            volume_variances.append(volume_variance)
        volumes = np.array(volumes).flatten()
        volume_variances = np.array(volume_variances).flatten()
        min_idx = np.argmin(volumes)
        max_idx = np.argmax(volumes)
        systole_var.append(volume_variances[min_idx])
        diastole_var.append(volume_variances[max_idx])
        systole_mean.append(volumes[min_idx])
        diastole_mean.append(volumes[max_idx])

    # STAGE3 apply volume variance to generate distribution and save to csv file
    convert_to_csv(diastole_mean, systole_mean, submit_csv_path=file_path,
                   diastole_std=2*np.sqrt(diastole_var),systole_std=1.5*np.sqrt(systole_var))
    
def fusion_submit(stage2_data_root_dir, test_data_root_dir,
                  adapters, fusion_area_fn, fusion_test_fn, 
                  fixed_size, file_path):
    # generate area data and label
    train_patch, train_label, val_patch, val_label = getData(stage2_data_root_dir, fixed_size)
    pred_areas = []
    length = 2000
    for i in range(len(train_patch))[:length]:
        if i % 500 == 0:
            print(i)
        x_e = train_patch[i].astype('float32')[None, None, :, :]
        preds = []
        for adapter in adapters:
            preds.append(adapter.convert(x_e))
        pred_e = np.concatenate(preds, axis=1)
        area_e = fusion_area_fn(pred_e)
        pred_areas.extend(area_e)
    gt_areas = train_label.mean(1).mean(1)[:length]
    pred_areas = np.array(pred_areas).flatten()
    
    # compute variance lookup table
    assert len(pred_areas) == len(gt_areas)
    var_lookup_table = varianceLUT(pred_areas, gt_areas)

    # FUSION read test patch, location and resolution
    x_data_test, location_data_test, resolution_data_test = utee.load_patch_stacks(test_data_root_dir, fixed_size)
       
    # FUSION make prediction and compute volume variance
    systole_mean = []
    diastole_mean = []
    systole_var = []
    diastole_var = []
    stage3_n_test_batches = len(x_data_test)
    for i in range(stage3_n_test_batches):
        if i % 50 == 0:
            print(i)
        n_samples = len(x_data_test[i])
        volumes = []
        volume_variances = []
        for j in range(n_samples):
            x_e = x_data_test[i][j]
            preds = []
            for adapter in adapters:
                preds.append(adapter.convert(x_e))
            pred_e = np.concatenate(preds, axis=1)
            location_e = location_data_test[i][j]
            resolution_e = resolution_data_test[i][j]
            area, volume = fusion_test_fn(pred_e, location_e, resolution_e)
            volume_variance = volumeVar(area.flatten(), location_e, resolution_e, fixed_size, var_lookup_table)
            volumes.append(volume)
            volume_variances.append(volume_variance)
        volumes = np.array(volumes).flatten()
        volume_variances = np.array(volume_variances).flatten()
        min_idx = np.argmin(volumes)
        max_idx = np.argmax(volumes)
        systole_var.append(volume_variances[min_idx])
        diastole_var.append(volume_variances[max_idx])
        systole_mean.append(volumes[min_idx])
        diastole_mean.append(volumes[max_idx])
        
    # save to submit
    convert_to_csv(diastole_mean, systole_mean, submit_csv_path=file_path,
                   diastole_std=np.sqrt(diastole_var),systole_std=np.sqrt(systole_var))