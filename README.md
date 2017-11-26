Paper link: https://arxiv.org/abs/1702.03833 

please cite the paper if you find this project useful for your research

OS: linux

Hardware: NVIDIA GPU required

3rd-party software: CUDA, caffe, opencv, theano, lasagne, matlab, and other dependencies required by them. We use the bleeding edge edition of theano, lasagne and caffe.

# HOW TO TRAIN

Firstly, specify the path to target folder in SETTINGS.json:

1. **IN_TRAIN_DATA_PATH**, path to train folder, please put in it `raw` folder
2. **IN_VALIDATE_DATA_PATH**, path to validate folder, please put it in `raw` folder
3. **OUT_TRAIN_DATA_PATH**, path to which the preprocess results of train folder are stored, please put it in `clean` folder
4. **OUT_VALIDATE_DATA_PATH**, path to which the preprocess results of validate folder are stored, please put it in `clean` floder
5. **TRAIN_LABEL_PATH**, path to train.csv, in which the number of rows should equal to the numpy of subfolders of IN_TRAIN_DATA_PATH
6. **VAL_LABEL_PATH**, path to validate.csv
7. FUSION_SNAPSHOT_PATH, path to model snapshot, **please do not modify this**



This algorithm includes two stages, the first one is detecting the heart area, which requires opencv, matlab and caffe. The parameters are already stored in folder */stage1*, to re-train those parameters by your own, please refer to the README.md in stage1. The output of this stage is a stack of images at per frame, and stored in disk.

The second stage is computing volume out of a patch stack. It is an ensamble consists of 6 models. Each model is packaged by an adaptor class in directory fusion(fcn1, fcn2...). Our auto-learning procedure learning the weight of each model in ensamble. And it will automatically gives the submit result.

Then, run `python train.py`

# How to predict

1. **IN_TEST_DATA_PATH**, path to test folder
2. **OUT_TEST_DATA_PATH**, path to which the preprocess results of test folder are stored
3. **SUBMIT_PATH**, path to save the submit csv file

Place the new test data in **IN_TEST_DATA_PATH**, also specify **OUT_TEST_DATA_PATH** and **SUBMIT_PATH**

Then, run `python predict.py`
