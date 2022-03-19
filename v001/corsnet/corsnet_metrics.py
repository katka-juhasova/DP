import tensorflow as tf


def rmse_t(y_true, y_pred):
    y_true_t = y_true[:, :-1, -1]
    y_pred_t = y_pred[:, :-1, -1]

    return tf.sqrt(tf.reduce_mean(tf.square(y_true_t - y_pred_t)))


def rmse_r(y_true, y_pred):
    y_true_r = y_true[:, :-1, :-1]
    y_pred_r = y_pred[:, :-1, :-1]

    return tf.sqrt(tf.reduce_mean(tf.square(y_true_r - y_pred_r)))
