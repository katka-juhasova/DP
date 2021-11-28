import h5py
import matplotlib.pyplot as plt


data_path = 'data/modelnet40_ply_hdf5_2048/train0.h5'

f = h5py.File(data_path, 'r')
data = f['data'][:]
label = f['label'][:]

print(label)

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

plt.show()
