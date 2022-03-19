import numpy as np
import tensorflow as tf
from .utils import load_h5
from .utils import shuffle_data


class Generator(tf.keras.utils.Sequence):
    def __init__(self, filename, batch_size=32, jitter=True,
                 shuffle=True, shuffle_points=True):
        self.filename = filename
        self.batch_size = batch_size
        self.jitter = jitter
        self.shuffle = shuffle
        self.shuffle_points = shuffle_points
        self.src, self.temp, self.t_matrix = load_h5(filename)

    @staticmethod
    def _jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. Jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        b, n, c = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(
            sigma * np.random.randn(b, n, c), -1 * clip, clip)
        jittered_data += batch_data

        return jittered_data

    @staticmethod
    def _shuffle_point_cloud(batch_data):
        """ Randomly shuffle points within each point cloud.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, batch of shuffled point clouds
        """
        idx = np.arange(batch_data.shape[1])
        np.random.shuffle(idx)

        return batch_data[:, idx, :]

    def on_epoch_end(self):
        if self.shuffle:
            self.src, self.temp, self.t_matrix, _ = shuffle_data(
                self.src, self.temp, self.t_matrix)

    def __len__(self):
        n = self.src.shape[0]
        return n // self.batch_size

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_src = self.src[start_idx:end_idx, :, :]
        batch_temp = self.temp[start_idx:end_idx, :, :]
        batch_t_matrix = self.t_matrix[start_idx:end_idx, :, :]

        if self.jitter:
            batch_src = self._jitter_point_cloud(batch_src)
            batch_temp = self._jitter_point_cloud(batch_temp)

        if self.shuffle_points:
            batch_src = self._shuffle_point_cloud(batch_src)
            batch_temp = self._shuffle_point_cloud(batch_temp)

        # Returns [p_src, p_tmp], g_gt
        return [batch_src, batch_temp], batch_t_matrix
