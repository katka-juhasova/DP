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
    data = f['data'][:]
    label = f['label'][:]

    return data, label


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx
