import argparse
import os
import numpy as np
import h5py
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, required=True,
                    help='Src dir with data to preprocess')
parser.add_argument('--dest_dir', type=str, required=True,
                    help='Dest dir for preprocessed data')
parser.add_argument('--same_points', type=bool, default=False,
                    help='Setting for src and temp point cloud sampling. \
                    If True both point clouds consist of the same points \
                    [default: False]')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024] [default: 1024]')
parser.add_argument('--r_min', type=float, default=-10.0,
                    help='Lower limit for rotation in degrees [default: -10.0]')
parser.add_argument('--r_max', type=float, default=10.0,
                    help='Upper limit for rotation in degrees [default: 10.0]')
parser.add_argument('--t_min', type=float, default=-0.4,
                    help='Lower limit for translation [default: -0.4]')
parser.add_argument('--t_max', type=float, default=0.4,
                    help='Upper limit for translation [default: 0.4]')
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


src_file = os.path.join(SRC_DIR, 'symbols.h5')
new_dir = 'Symbols_same_{same_points}_r_{r_min}_{r_max}_t_{t_min}_{t_max}'
new_dir = new_dir.format(same_points=int(SAME_POINTS),
                         r_min=R_MIN, r_max=R_MAX,
                         t_min=T_MIN, t_max=T_MAX)
dest_train_file = os.path.join(DEST_DIR, new_dir, 'train_dataset.h5')
dest_test_file = os.path.join(DEST_DIR, new_dir, 'test_dataset.h5')

os.makedirs(os.path.join(DEST_DIR, new_dir), exist_ok=True)

# Dataset shape is [n, 2048, 3]
# Load data
symbols = utils.load_h5(src_file)

if SAME_POINTS:
    src_p = symbols[:, :NUM_POINT, :]
    temp_p = symbols[:, :NUM_POINT, :]
else:
    # Create (source, temp) pairs by splitting each point cloud
    # of size 2048x3 into 2 point clouds 1024x3
    src_p = symbols[:, :NUM_POINT, :]
    temp_p = symbols[:, NUM_POINT:2 * NUM_POINT, :]

n_symbols = symbols.shape[0]
# Random rotation from range [R_MIN, R_MAX] degrees
r = utils.random_rotation(n=n_symbols, r_min=R_MIN, r_max=R_MAX)
# Random translation from range [T_MIN, T_MAX]
t = utils.random_translation(n=n_symbols, t_min=T_MIN, t_max=T_MAX)
# Last row for transformation matrix
last_row = np.array([0., 0., 0., 1.])
last_row = np.tile(last_row, (n_symbols, 1, 1))
# Build transformation matrix
t_matrix = np.concatenate((r, t), axis=2)
t_matrix = np.concatenate((t_matrix, last_row), axis=1)

ones_col = np.ones((temp_p.shape[0], NUM_POINT, 1))
temp_p = np.concatenate((temp_p, ones_col), axis=2)

# Apply transformation
temp_p = np.array(
    [temp_p[i, :, :] @ t_matrix[i, :, :].T for i in range(temp_p.shape[0])]
)
# Remove last column with ones
temp_p = temp_p[:, :, :-1]

# Split data: 80% train set, 20% test set
split_idx = int(n_symbols * 0.8)
train_src, test_src = np.split(src_p, [split_idx])
train_temp, test_temp = np.split(temp_p, [split_idx])
train_t_matrix, test_t_matrix = np.split(t_matrix, [split_idx])

# Save the datasets
with h5py.File(dest_train_file, 'w') as f:
    src_p1 = f.create_dataset('source', data=train_src)
    temp_p1 = f.create_dataset('template', data=train_temp)
    matrix_p1 = f.create_dataset('transform_matrix', data=train_t_matrix)

with h5py.File(dest_test_file, 'w') as f:
    src_p2 = f.create_dataset('source', data=test_src)
    temp_p2 = f.create_dataset('template', data=test_temp)
    matrix_p2 = f.create_dataset('transform_matrix', data=test_t_matrix)

print('Train dataset successfully saved to {}'.format(dest_train_file))
print('Test dataset successfully saved to {}'.format(dest_test_file))
