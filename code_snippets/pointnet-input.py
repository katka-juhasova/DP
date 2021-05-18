import numpy as np
import h5py
import matplotlib.pyplot as plt


data_path = 'train0.h5'

f = h5py.File(data_path, 'r')
data = f['data'][:]
label = f['label'][:]

print(label)

n = 100

x = data[n,:,0]
y = data[n,:,1]
z = data[n,:,2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
