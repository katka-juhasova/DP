import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_POINTS = 1024
NUM_CLASSES = 40
BATCH_SIZE = 32


def conv1d_bn(x, filters, name=None):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("relu", name=name)(x)


def dense_bn(x, filters, name=None):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("relu", name=name)(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {
            "num_features": self.num_features,
            "l2reg": self.l2reg
        }


def tnet(inputs, num_features, regularize=False, name=None):
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())

    x = conv1d_bn(inputs, 64)
    x = conv1d_bn(x, 128)
    x = conv1d_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)

    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=OrthogonalRegularizer(
            num_features) if regularize else None
    )(x)

    feat_t = layers.Reshape((num_features, num_features))(x)

    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1), name=name)([inputs, feat_t])


def get_model(num_points=NUM_POINTS, num_classes=NUM_CLASSES,
              name="pointnet", weights=None):

    inputs = keras.Input(shape=(num_points, 3), name=name + "_input")

    x = tnet(inputs, 3, regularize=False)
    x = conv1d_bn(x, 64)
    x = conv1d_bn(x, 64)
    x = tnet(x, 64, regularize=True, name=name + "_local_features")
    x = conv1d_bn(x, 64)
    x = conv1d_bn(x, 128)
    x = conv1d_bn(x, 1024)
    x = layers.GlobalMaxPooling1D(name=name + "_global_features")(x)
    x = dense_bn(x, 512)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    if weights:
        model.load_weights(weights)

    return model
