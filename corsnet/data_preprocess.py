import os
import shutil
import h5py
import numpy as np
import pointnet.utils as utils
from scipy.spatial.transform import Rotation as R


DATA_DIR = 'data/modelnet40_ply_hdf5_2048'
TRAIN_FILES = os.path.join(DATA_DIR, 'train_files.txt')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

NEW_DATA_DIR = 'data/CorsNet'
NEW_TRAIN_FILES = os.path.join(NEW_DATA_DIR, 'train_files.txt')
NEW_TEST_FILES = os.path.join(NEW_DATA_DIR, 'test_files.txt')

# Copy lists of train and test files
shutil.copy(TRAIN_FILES, NEW_TRAIN_FILES)
shutil.copy(TEST_FILES, NEW_TEST_FILES)

# Read the lists of train and test files
train_files = utils.get_data_files(TRAIN_FILES)
test_files = utils.get_data_files(TEST_FILES)

NUM_POINTS = 1024

# ModelNet40 dataset shape is [n, 2048, 3]
# Create (source, template) pairs by splitting each point cloud of size
# 2048x3 into 2 point clouds 1024x3
for file in train_files + test_files:
    new_file = os.path.join(NEW_DATA_DIR, os.path.basename(file))
    print('Processing file {}'.format(new_file))

    data, _ = utils.load_h5(file)
    src_p = data[:, :NUM_POINTS, :]
    temp_p = data[:, :NUM_POINTS, :]

    # TODO: build temp_p with second half of the original point cloud
    # temp_p = data[:, NUM_POINTS:, :]

    # Random rotation from range [0, 45] degrees around arbitrarily chosen axis
    axis = np.random.choice(['x', 'y', 'z'], data.shape[0])
    degree = np.random.uniform(0., 90., data.shape[0])
    r = np.array(
        [R.from_euler(axis[i], degree[i], degrees=True).as_matrix()
         for i in range(data.shape[0])]
    )
    # Random translation from range [0, 0.8]
    t = np.random.uniform(0., 0.8, (data.shape[0], 3, 1))
    # Last row for transformation matrix
    last_row = np.array([0., 0., 0., 1.])
    last_row = np.tile(last_row, (data.shape[0], 1, 1))
    # Build transformation matrix
    t_matrix = np.concatenate((r, t), axis=2)
    t_matrix = np.concatenate((t_matrix, last_row), axis=1)

    ones_col = np.ones((temp_p.shape[0], NUM_POINTS, 1))
    temp_p = np.concatenate((temp_p, ones_col), axis=2)

    temp_p = np.array(
        [temp_p[i, :, :] @ t_matrix[i, :, :].T for i in range(temp_p.shape[0])]
    )
    # Remove last column with ones
    temp_p = temp_p[:, :, :-1]

    with h5py.File(new_file, 'w') as f:
        src = f.create_dataset('source', data=src_p)
        temp = f.create_dataset('template', data=temp_p)
        matrix = f.create_dataset('transform_matrix', data=t_matrix)
