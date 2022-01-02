import numpy as np
import tensorflow as tf
import pointnet.utils as utils


class Generator(tf.keras.utils.Sequence):
    def __init__(self, filenames, num_point=1024, batch_size=32,
                 jitter=True, rotate=True, shuffle=True):
        self.filenames = filenames
        self.num_point = num_point
        self.batch_size = batch_size
        self.jitter = jitter
        self.rotate = rotate
        self.shuffle = shuffle

        self.data = list()
        self.labels = list()
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
            data, _ = utils.load_h5(file)
            file_sizes.append(data.shape[0])

        return file_sizes

    @staticmethod
    def _rotate_point_cloud(batch_data):
        """ Randomly rotate the point clouds to augment the dataset
              rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)),
                                          rotation_matrix)

        return rotated_data

    @staticmethod
    def _jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points, jittering is per point.
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
            self.data, self.labels = utils.load_h5(file)
            self.data = self.data[:, 0:self.num_point, :]

            if self.shuffle:
                self.data, self.labels, _ = utils.shuffle_data(self.data,
                                                               self.labels)
            self.labels = np.squeeze(self.labels)

            self.last_file_idx = file_idx

        start_idx = tmp_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_data = self.data[start_idx:end_idx, :, :]
        batch_labels = self.labels[start_idx:end_idx]

        if self.rotate:
            batch_data = self._rotate_point_cloud(batch_data)

        if self.jitter:
            batch_data = self._jitter_point_cloud(batch_data)

        return batch_data, batch_labels
