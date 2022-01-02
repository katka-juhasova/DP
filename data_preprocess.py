import os
import argparse
import shutil
import pointnet.utils as utils
import numpy as np
from scipy.spatial.transform import Rotation as R
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, default='modelnet40_ply_hdf5_2048',
                    help='Src dir with data to preprocess \
                    [default: modelnet40_ply_hdf5_2048]')
parser.add_argument('--dest_dir', type=str, default='CorsNet',
                    help='Dest dir for preprocessed data [default: CorsNet]')
parser.add_argument('--same_points', type=bool, default=False,
                    help='Setting for src and temp point cloud sampling. \
                    If True both point clouds are consist of the same points \
                    [default: False]')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--r_min', type=float, default=0.0,
                    help='Lower limit for rotation in degrees [default: 0.0]')
parser.add_argument('--r_max', type=float, default=45.0,
                    help='Upper limit for rotation in degrees [default: 45.0]')
parser.add_argument('--t_min', type=float, default=0.0,
                    help='Lower limit for translation [default: 0.0]')
parser.add_argument('--t_max', type=float, default=0.8,
                    help='Upper limit for translation [default: 0.8]')
args = parser.parse_args()


# CONFIG
SRC_DIR = args.src_dir
DEST_DIR = args.dest_dir
SAME_POINTS = args.same_points
NUM_POINT = args.num_point
R_MIN = args.r_min
R_MAX = args.r_max
T_MIN = args.t_min
T_MAX = args.t_max

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', SRC_DIR)
TRAIN_FILES = os.path.join(DATA_DIR, 'train_files.txt')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

NEW_DATA_DIR = os.path.join(BASE_DIR, 'data', DEST_DIR)
NEW_TRAIN_FILES = os.path.join(NEW_DATA_DIR, 'train_files.txt')
NEW_TEST_FILES = os.path.join(NEW_DATA_DIR, 'test_files.txt')

# Copy lists of train and test files
shutil.copy(TRAIN_FILES, NEW_TRAIN_FILES)
shutil.copy(TEST_FILES, NEW_TEST_FILES)

# Read the lists of train and test files
train_files = utils.get_data_files(TRAIN_FILES)
test_files = utils.get_data_files(TEST_FILES)


# ModelNet40 dataset shape is [n, 2048, 3]
for file in train_files + test_files:
    new_file = os.path.join(NEW_DATA_DIR, os.path.basename(file))
    print('Processing file {}'.format(new_file))

    data, _ = utils.load_h5(file)

    if SAME_POINTS:
        src_p = data[:, :NUM_POINT, :]
        temp_p = data[:, :NUM_POINT, :]
    else:
        # Create (source, temp) pairs by splitting each point cloud
        # of size 2048x3 into 2 point clouds 1024x3
        src_p = data[:, :NUM_POINT, :]
        temp_p = data[:, NUM_POINT:2 * NUM_POINT, :]

    # TODO: temp points normalizing into unit box at the origin [0,1]^3
    # Random rotation from range [R_MIN, R_MAX] degrees
    r_degrees = np.random.uniform(R_MIN, R_MAX, size=(data.shape[0], 3))
    r = np.array(
        [R.from_euler('xyz', degree, degrees=True).as_matrix()
         for degree in r_degrees]
    )
    # Random translation from range [T_MIN, T_MAX]
    t = np.random.uniform(0., 0.8, (data.shape[0], 3, 1))
    # Last row for transformation matrix
    last_row = np.array([0., 0., 0., 1.])
    last_row = np.tile(last_row, (data.shape[0], 1, 1))
    # Build transformation matrix
    t_matrix = np.concatenate((r, t), axis=2)
    t_matrix = np.concatenate((t_matrix, last_row), axis=1)

    ones_col = np.ones((temp_p.shape[0], NUM_POINT, 1))
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
