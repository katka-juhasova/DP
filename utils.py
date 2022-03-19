import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_h5(filename):
    f = h5py.File(filename, 'r')
    data = f['symbols'][:]
    return data


def random_rotation(n, r_min, r_max):
    """Returns N random 3D rotations from range [r_min, r_max] degrees."""
    r_degrees = np.random.uniform(r_min, r_max, size=(n, 3))
    r = np.array(
        [R.from_euler('xyz', degree, degrees=True).as_matrix()
         for degree in r_degrees]
    )
    return r


def random_translation(n, t_min, t_max):
    """Returns N random 3D translations from range [t_min, t_max]."""
    t = np.random.uniform(t_min, t_max, (n, 3, 1))
    return t


