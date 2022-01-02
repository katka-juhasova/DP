import os
import h5py
import numpy as np


def get_data_files(files_list):
    with open(files_list) as f:
        data_files = f.read().splitlines()

    dir_path = os.path.dirname(files_list)
    data_files = [os.path.join(dir_path, file) for file in data_files]

    return data_files


def load_h5(filename):
    f = h5py.File(filename, 'r')
    src = f['source'][:]
    temp = f['template'][:]
    t_matrix = f['transform_matrix'][:]

    return src, temp, t_matrix


def shuffle_data(src, temp, t_matrix):
    idx = np.arange(len(src))
    np.random.shuffle(idx)

    return src[idx, ...], temp[idx, ...], t_matrix[idx, ...], idx
