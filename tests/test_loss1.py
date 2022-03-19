import tensorflow as tf
from corsnet.losses import CorsNetLoss1


def test_loss1():
    y_true = tf.Variable([[[1., 0., 0., 1.],
                           [0., 3., 0., 1.],
                           [0., 0., 5., 1.],
                           [0., 0., 0., 1.]],

                          [[1., 0., 0., 1.],
                           [0., 3., 0., 1.],
                           [0., 0., 5., 1.],
                           [0., 0., 0., 1.]]
                          ])
    y_pred = tf.Variable([[[1., 0., 0., 1.],
                           [0., 1., 0., 1.],
                           [0., 0., 1., 1.],
                           [0., 0., 0., 1.]],

                          [[1., 0., 0., 1.],
                           [0., 1., 0., 1.],
                           [0., 0., 1., 1.],
                           [0., 0., 0., 1.]]
                          ])

    l = CorsNetLoss1()
    assert tf.equal(l(y_true, y_pred), tf.constant(20.))
