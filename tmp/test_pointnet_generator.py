import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'pointnet'))
import pointnet_utils as utils
from pointnet_generator import Generator


DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'modelnet40_ply_hdf5_2048')
TRAIN_FILES = os.path.join(DATA_DIR, 'train_files.txt')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

train_files = utils.get_data_files(TRAIN_FILES)
test_files = utils.get_data_files(TEST_FILES)

for file in train_files:
    x, _ = utils.load_h5(file)
    print('{} point cloud num: {}'.format(file, x.shape[0]))

for file in test_files:
    x, _ = utils.load_h5(file)
    print('{} point cloud num: {}'.format(file, x.shape[0]))

train_generator = Generator(train_files)
test_generator = Generator(test_files)
print(len(train_generator))
print(len(test_generator))

data, labels = test_generator.__getitem__(13)
print(test_generator.last_file_idx == test_generator.filename_idxs[0])
