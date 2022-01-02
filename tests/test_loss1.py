import os
import sys
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'corsnet'))
from corsnet_loss1 import CorsNetLoss1


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
