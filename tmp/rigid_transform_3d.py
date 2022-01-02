#!/usr/bin/python

import numpy as np
from scipy.spatial.transform import Rotation as Rotation


# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector


def rigid_transform_3D(A, B):
    # Find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # Ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # Subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # Find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    Vt[2, :] *= -1

    t = -R @ centroid_A + centroid_B

    return R, t


if __name__ == '__main__':
    A = np.array([[0., 0., 0.],
                  [1., 0., 0.],
                  [0., 0., 1.],
                  [1., 0., 1.],
                  [0., 1., 0.],
                  [1., 1., 0.],
                  [0., 1., 1.],
                  [1., 1., 1.]])

    R = Rotation.random().as_matrix()
    # Random translation
    t = np.random.randn(3, 1)
    # Build transformation matrix
    t_matrix = np.concatenate((R, t), axis=1)
    t_matrix = np.concatenate((t_matrix, np.array([[0., 0., 0., 1.]])))

    ones_col = np.ones((A.shape[0], 1))
    A = np.concatenate((A, ones_col), axis=1)
    B = A.dot(t_matrix.T)
    # Remove last column with ones
    A = A[:, :-1]
    B = B[:, :-1]

    ret_R, ret_t = rigid_transform_3D(A.T, B.T)
    B2 = (ret_R @ A.T) + ret_t
    error = np.sum(B - B2.T)
    print('Success!')
