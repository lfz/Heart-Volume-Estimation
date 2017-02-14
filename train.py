
# coding: utf-8

# In[1]:

import json
from stage1.breakRedundancy import breakRedundancy 
from stage1.getStack import preparePatchStack
from stage1.savePatchBatch import saveGroundTruthPatchs
import utee
from fusion_train import fusion_train


# In[2]:

with open('./SETTINGS.json', 'r') as f:
    config = json.load(f)

in_train_data_path = config['IN_TRAIN_DATA_PATH']
in_val_data_path = config['IN_VALIDATE_DATA_PATH']
out_train_data_path = config['OUT_TRAIN_DATA_PATH']
out_val_data_path = config['OUT_VALIDATE_DATA_PATH']
train_label_path = config['TRAIN_LABEL_PATH']
val_label_path = config['VAL_LABEL_PATH']
clean_root = './clean'

# break redundancy inplace
print("Breaking redundancy of {} ".format(in_train_data_path))
breakRedundancy(in_train_data_path)
print("Breaking redundancy of {} ".format(in_val_data_path))
breakRedundancy(in_val_data_path)

# convert to inner data format
print("Converting training data from {} to {}/min and {}/max".format(in_train_data_path, clean_root, clean_root))
saveGroundTruthPatchs(in_train_data_path, train_label_path, clean_root)
print("Converting validatin data from {} to {}".format(in_val_data_path, out_val_data_path))
preparePatchStack(in_val_data_path, out_val_data_path)

# load train and val csv data
fixed_size = (48, 48)
volume_data_train = utee.read_csv(train_label_path)
volume_data_val = utee.read_csv(val_label_path)
x_data_train, location_data_train, resolution_data_train = utee.load_min_max_patch_stacks(clean_root, fixed_size)
x_data_val, location_data_val, resolution_data_val = utee.load_patch_stacks(out_val_data_path, fixed_size)
print("loaded x_data_train: {}, location_data_train: {}, resolution_data_train: {}".format(
    len(x_data_train), len(location_data_train), len(resolution_data_train)))
print("loaded x_data_val: {}, location_data_val: {}, resolution_data_val: {}".format(
    len(x_data_val), len(location_data_val), len(resolution_data_val)))


fusion_train(x_data_train, location_data_train, resolution_data_train, volume_data_train,
                 x_data_val, location_data_val, resolution_data_val, volume_data_val[8:],
                 fixed_size, 100, 0.005)

