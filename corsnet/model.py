import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


NUM_POINT = 1024


def conv1d_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("relu")(x)


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
        # NOTE: no need to handle special reflection case,
        # since we don't support mirror transformation

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


def get_model(num_point=NUM_POINT, name='corsnet'):
    # Source PointNet (without T-Net)
    input_src = keras.Input(shape=(num_point, 3))
    x_src = conv1d_bn(input_src, 64)
    loc_feat_src = conv1d_bn(x_src, 64)
    x_src = conv1d_bn(loc_feat_src, 64)
    x_src = conv1d_bn(x_src, 128)
    x_src = conv1d_bn(x_src, 1024)
    glob_feat_src = layers.GlobalMaxPooling1D()(x_src)

    # Template PointNet (without T-Net)
    input_temp = keras.Input(shape=(num_point, 3))
    x_temp = conv1d_bn(input_temp, 64)
    loc_feat_temp = conv1d_bn(x_temp, 64)
    x_temp = conv1d_bn(loc_feat_temp, 64)
    x_temp = conv1d_bn(x_temp, 128)
    x_temp = conv1d_bn(x_temp, 1024)
    glob_feat_temp = layers.GlobalMaxPooling1D()(x_temp)

    # Correspondence estimation
    glob_feat_src = layers.RepeatVector(n=1024)(glob_feat_src)
    glob_feat_temp = layers.RepeatVector(n=1024)(glob_feat_temp)

    cors = layers.Concatenate()([loc_feat_src, glob_feat_temp, glob_feat_src])
    cors = conv1d_bn(cors, 512)
    cors = conv1d_bn(cors, 256)
    cors = conv1d_bn(cors, 128)
    cors = conv1d_bn(cors, 3)

    # SVD
    g_est = SVDLayer()([cors, input_src])

    return keras.Model(inputs=[input_src, input_temp], outputs=g_est,
                       name=name)


# Function for callback for modifying the learning rate
# NOTE: to be exact, it should be 74, 149 and 199
def lr_scheduler(epoch, lr):
    if epoch in (75, 150, 200):
        return tf.divide(lr, 10.)
    else:
        return lr
