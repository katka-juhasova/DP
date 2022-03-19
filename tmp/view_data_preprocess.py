import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '..', 'data', 'SymbolsDataset',
                         'Symbols_same_0_r_0_45_t_0_0.8', 'train_dataset.h5')
f = h5py.File(data_path, 'r')

src = f['source'][:]
temp = f['template'][:]
t_matrix = f['transform_matrix'][:]

N = 120
NUM_POINTS = 1024

x1 = src[N, :, 0]
y1 = src[N, :, 1]
z1 = src[N, :, 2]

x2 = temp[N, :, 0]
y2 = temp[N, :, 1]
z2 = temp[N, :, 2]

# Transform source point cloud
ones_col = np.ones((NUM_POINTS, 1))
ext_src = np.concatenate((src[N], ones_col), axis=1)
transformed = ext_src @ t_matrix[N].T
transformed = transformed[:, :-1]

x3 = transformed[:, 0]
y3 = transformed[:, 1]
z3 = transformed[:, 2]

# Get rotation in degrees about each of the axis
r = R.from_matrix(t_matrix[N, :-1, :-1]).as_euler('xyz', degrees=True)
# Get translation
t = t_matrix[N][:, -1]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, y1, z1, label='source')
ax.scatter(x2, y2, z2, label='template')
ax.scatter(x3, y3, z3, label='transformed')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('Rotation: {}\n Translation: {}'.format(r, t))
plt.legend(loc='upper right')
plt.show()
