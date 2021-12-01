import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pointnet.pointnet as pointnet


NUM_POINTS = 1024


class SVDLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SVDLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        cors, p_src = inputs
        p_pred = tf.add(p_src, cors)

        # Find mean column wise
        src_centroid = tf.reduce_mean(p_src, axis=1)
        src_centroid = tf.expand_dims(src_centroid, axis=1)
        pred_centroid = tf.reduce_mean(p_pred, axis=1)
        pred_centroid = tf.expand_dims(pred_centroid, axis=1)

        # Subtract mean
        src_trans = tf.subtract(p_src, src_centroid)
        pred_trans = tf.subtract(p_pred, pred_centroid)

        # h = covariance matrix
        h = tf.matmul(tf.transpose(src_trans, perm=[0, 2, 1]), pred_trans)
        s, u, v = tf.linalg.svd(h)

        # Rotation
        r_est = tf.matmul(v, tf.transpose(u, perm=[0, 2, 1]))

        # TODO: implement handling of the sign problem
        # # handle svd sign problem
        # #  https://medium.com/machine-learning-world/linear-algebra-points-matching-with-svd-in-3d-space-2553173e8fed
        # # Special reflection case
        # if tf.less(tf.linalg.det(r_est), 0.):
        #     slice1 = v[:, :, :2]
        #     slice2 = tf.scalar_mul(-1.0, v[:, :, 2])
        #     slice2 = tf.expand_dims(slice2, axis=-1)
        #     v = tf.concat([slice1, slice2], axis=-1)
        #     r_est = tf.matmul(v, tf.transpose(u, perm=[0, 2, 1]))

        # Translation
        t_est = - tf.matmul(r_est, tf.transpose(src_centroid, perm=[0, 2, 1]))
        t_est = t_est + tf.transpose(pred_centroid, perm=[0, 2, 1])

        # Concatenate rotation and translation matrices
        g_est = tf.concat([r_est, t_est], axis=2)

        # Add row to matrix so that it's of shape 4x4
        batch_size = tf.shape(g_est)[0]
        g_est_last = tf.constant([[[0., 0., 0., 1.]]])
        g_est_last = tf.broadcast_to(g_est_last, shape=(batch_size, 1, 4))
        g_est = tf.concat([g_est, g_est_last], axis=1)

        return g_est


def get_model(num_points=NUM_POINTS, pointnet_weights=None, name='corsnet'):
    # Source PointNet
    src_pointnet = pointnet.get_model(num_points=num_points,
                                      name='src_pointnet')

    input_src_layer = 'src_pointnet_input'
    loc_feat_src_layer = 'src_pointnet_local_features'
    glob_feat_src_layer = 'src_pointnet_global_features'
    input_src = src_pointnet.get_layer(input_src_layer).output
    loc_feat_src = src_pointnet.get_layer(loc_feat_src_layer).output
    glob_feat_src = src_pointnet.get_layer(glob_feat_src_layer).output

    # Template PointNet
    temp_pointnet = pointnet.get_model(num_points=num_points,
                                       name='temp_pointnet')

    input_temp_layer = 'temp_pointnet_input'
    glob_feat_temp_layer = 'temp_pointnet_global_features'
    input_temp = temp_pointnet.get_layer(input_temp_layer).output
    glob_feat_temp = temp_pointnet.get_layer(glob_feat_temp_layer).output

    # Add load weights to PointNet models and set trainable=False
    if pointnet_weights:
        src_pointnet.load_weights(pointnet_weights)
        temp_pointnet.load_weights(pointnet_weights)
        src_pointnet.trainable = False
        temp_pointnet.trainable = False

    # Correspondence estimation
    glob_feat_src = layers.RepeatVector(n=1024)(glob_feat_src)
    glob_feat_temp = layers.RepeatVector(n=1024)(glob_feat_temp)

    cors = layers.Concatenate()([loc_feat_src, glob_feat_temp, glob_feat_src])
    cors = pointnet.conv1d_bn(cors, 512)
    cors = pointnet.conv1d_bn(cors, 256)
    cors = pointnet.conv1d_bn(cors, 128)
    cors = pointnet.conv1d_bn(cors, 3)

    # SVD
    g_est = SVDLayer()([cors, input_src])

    return keras.Model(inputs=[input_src, input_temp], outputs=g_est,
                       name=name)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weights = os.path.join(
        BASE_DIR,
        (r"../models/2021-10-27_07:38:35_PointNet-1zy4zmyd/"
         r"model.epoch239-loss2.17-acc0.95-val_loss2.63-val_acc0.87.h5")
    )

    model = get_model(num_points=NUM_POINTS, pointnet_weights=weights)
    model.summary()
