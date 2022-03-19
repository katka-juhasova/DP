import numpy as np
import tensorflow as tf
import v001.corsnet.corsnet_utils as utils


class Generator(tf.keras.utils.Sequence):
    def __init__(self, filenames, batch_size=32, jitter=True,
                 shuffle=True, shuffle_points=True):
        self.filenames = filenames
        self.batch_size = batch_size
        self.jitter = jitter
        self.shuffle = shuffle
        self.shuffle_points = shuffle_points

        self.src = list()
        self.temp = list()
        self.t_matrix = list()
        self.filename_idxs = self._get_idxs()
        self.file_sizes = self._get_file_sizes()
        self.last_file_idx = -1

    def _get_idxs(self):
        filename_idxs = np.arange(0, len(self.filenames))

        if self.shuffle:
            np.random.shuffle(filename_idxs)

        return filename_idxs

    def _get_file_sizes(self):
        file_sizes = list()

        for file in self.filenames:
            src, _, _ = utils.load_h5(file)
            file_sizes.append(src.shape[0])

        return file_sizes

    @staticmethod
    def _jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(
            sigma * np.random.randn(B, N, C), -1 * clip, clip
        )
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

    def __len__(self):
        batch_num = 0
        for size in self.file_sizes:
            batch_num += size // self.batch_size

        return batch_num

    def __getitem__(self, index):
        file_idx = 0
        tmp_idx = index

        # Loop to determine file index
        while (
            file_idx < len(self.filenames)
            and (self.file_sizes[self.filename_idxs[file_idx]]
                 // self.batch_size <= tmp_idx)
        ):
            tmp_idx -= (self.file_sizes[self.filename_idxs[file_idx]]
                        // self.batch_size)
            file_idx += 1

        # If the file is already loaded, select slice,
        # otherwise load new file and slice
        if self.last_file_idx != file_idx:
            file = self.filenames[self.filename_idxs[file_idx]]
            self.src, self.temp, self.t_matrix = utils.load_h5(file)

            if self.shuffle:
                self.src, self.temp, self.t_matrix, _ = utils.shuffle_data(
                    self.src, self.temp, self.t_matrix)

            self.last_file_idx = file_idx

        start_idx = tmp_idx * self.batch_size
        end_idx = start_idx + self.batch_size
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
