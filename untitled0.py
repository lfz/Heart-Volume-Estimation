import json
from stage1.breakRedundancy import breakRedundancy 
from stage1.getStack import preparePatchStack
from stage1.savePatchBatch import saveGroundTruthPatchs
import utee
from fusion_train import fusion_train
from fusion_predict import fusion_predict


with open('./SETTINGS.json', 'r') as f:
    config = json.load(f)

in_test_data_path = config['IN_TEST_DATA_PATH']
out_test_data_path = config['OUT_TEST_DATA_PATH']
fusion_snapshot_path = config['FUSION_SNAPSHOT_PATH']
submisson_path = config['SUBMIT_PATH']

# print("Breadking redundancy of {}".format(in_test_data_path))
# breakRedundancy(in_test_data_path)
# print("Converting validatin data from {} to {}".format(in_test_data_path, out_test_data_path))
preparePatchStack(in_test_data_path, out_test_data_path)
fixed_size = (48, 48)


fusion_predict(fusion_snapshot_path, out_test_data_path, 'stage2', submisson_path, fixed_size)