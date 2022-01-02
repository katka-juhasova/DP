import os
import sys
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'corsnet'))
import metrics


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
                       [0., 1., 0., 0.],
                       [0., 0., 1., 1.],
                       [0., 0., 0., 1.]],

                      [[1., 0., 0., 1.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 1.],
                       [0., 0., 0., 1.]]
                      ])


def test_rmse_r():
    loss_r = metrics.rmse_r(y_true, y_pred)
    assert tf.equal(loss_r, tf.constant(1.4907119))


def test_rmse_t():
    loss_t = metrics.rmse_t(y_true, y_pred)
    assert tf.equal(loss_t, tf.constant(0.57735026))
