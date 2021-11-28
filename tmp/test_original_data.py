import os
import h5py
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR,
                         '../data/modelnet40_ply_hdf5_2048/train0.h5')

f = h5py.File(data_path, 'r')
data = f['data'][:]
label = f['label'][:]

N = 120
x = data[N, :, 0]
y = data[N, :, 1]
z = data[N, :, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('Label: ' + str(label[N, 0]))
plt.show()
