import tensorflow as tf


# https://github.com/hmgoforth/PointNetLK/blob/master/ptlk/pointlk.py#L43-L50
class CorsNetLoss1(tf.keras.losses.Loss):
    """
    Implementation of Loss1 function from
    https://ieeexplore.ieee.org/abstract/document/8978671.
    """
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        """
        :param y_true:
            matrix 4x4
                R_gt R_gt R_gt t_gt
                R_gt R_gt R_gt t_gt
                R_gt R_gt R_gt t_gt
                0.   0.   0.   1.

        :param y_pred:
            matrix 4x4
                R_est R_est R_est t_est
                R_est R_est R_est t_est
                R_est R_est R_est t_est
                0.    0.    0.    1.
        """
        y_pred_inv = tf.linalg.inv(y_pred)

        # G_est^(-1) . G_gt - I_4
        loss = (y_pred_inv @ y_true) - tf.eye(4)
        loss = tf.reduce_mean(tf.square(loss)) * 16

        return loss
