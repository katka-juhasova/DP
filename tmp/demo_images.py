import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


NUM_POINT = 1024
N = 120


def load_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    label = f['label'][:]

    return data, label


def registration_concept():
    filename = '../v001/data/ModelNet40/train0.h5'
    data, _ = load_h5(filename)

    src = data[N, :NUM_POINT, :]
    temp = data[N, NUM_POINT:, :]

    # Rotation & translation
    r = R.from_euler('xyz', [45., 0., 0.], degrees=True).as_matrix()
    t = np.array([[0.], [-0.3], [0.5]])
    # Last row for transformation matrix
    last_row = np.array([[0., 0., 0., 1.]])
    # Build transformation matrix
    t_matrix = np.concatenate((r, t), axis=-1)
    t_matrix = np.concatenate((t_matrix, last_row), axis=0)

    ones_col = np.ones((NUM_POINT, 1))
    temp = np.concatenate((temp, ones_col), axis=-1)

    trans = temp @ t_matrix.T
    # Remove last column with ones
    temp = temp[:, :-1]
    trans = trans[:, :-1]

    # Plot point clouds before registration
    plt.style.use('seaborn-deep')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(aspect=(1, 1, 1))

    ax.scatter(src[:, 0], src[:, 2], src[:, 1], alpha=0.35)
    ax.scatter([], [], [])
    ax.scatter(trans[:, 0], trans[:, 2], trans[:, 1], alpha=0.35)

    # Scene settings
    ax.set_axis_off()
    ax.view_init(0, 0)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # plt.show()
    plt.savefig('../docs/images/pre-registration.png')

    # Plot point clouds after registrtion
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(aspect=(1, 1, 1))

    ax.scatter(src[:, 0], src[:, 2], src[:, 1], alpha=0.35)
    ax.scatter([], [], [])
    ax.scatter(temp[:, 0], temp[:, 2], temp[:, 1], alpha=0.35)

    # Scene settings
    ax.set_axis_off()
    ax.view_init(0, 0)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # plt.show()
    plt.savefig('../docs/images/post-registration.png')


if __name__ == '__main__':
    registration_concept()
