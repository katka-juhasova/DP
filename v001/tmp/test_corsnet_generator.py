import os
import v001.corsnet.corsnet_utils as utils
from v001.corsnet.corsnet_generator import Generator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'CorsNet')
TRAIN_FILES = os.path.join(DATA_DIR, 'train_files.txt')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

train_files = utils.get_data_files(TRAIN_FILES)
test_files = utils.get_data_files(TEST_FILES)

for file in train_files:
    x, _, _ = utils.load_h5(file)
    print('{} point cloud num: {}'.format(file, x.shape[0]))

for file in test_files:
    x, _, _ = utils.load_h5(file)
    print('{} point cloud num: {}'.format(file, x.shape[0]))

train_generator = Generator(train_files)
test_generator = Generator(test_files)
print(len(train_generator))
print(len(test_generator))

[src, temp], t_matrix = test_generator.__getitem__(13)
print(src.shape, temp.shape, t_matrix.shape)
print(test_generator.last_file_idx == test_generator.filename_idxs[0])
