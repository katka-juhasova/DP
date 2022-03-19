import h5py
import numpy as np


def load_h5(filename: str):
    f = h5py.File(filename, 'r')
    src = f['source'][:]
    temp = f['template'][:]
    t_matrix = f['transform_matrix'][:]

    return src, temp, t_matrix


def shuffle_data(src, temp, t_matrix):
    idx = np.arange(len(src))
    np.random.shuffle(idx)

    return src[idx, ...], temp[idx, ...], t_matrix[idx, ...], idx
