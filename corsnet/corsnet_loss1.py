import tensorflow as tf


class CorsNetLoss1(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        """
        :param y_true:
            matrix 3x4
                R_gt R_gt R_gt t_gt
                R_gt R_gt R_gt t_gt
                R_gt R_gt R_gt t_gt

        :param y_pred:
            matrix 3x4
                R_est R_est R_est t_est
                R_est R_est R_est t_est
                R_est R_est R_est t_est
        """
        y_pred_inv = tf.linalg.inv(y_pred)

        loss = tf.linalg.matmul(y_pred_inv, y_true) - tf.eye(4)
        loss = tf.sqrt(tf.square(loss))

        return loss


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    y_true = tf.Variable([[[1., 0., 0., 1.],
                           [0., 3., 0., 1.],
                           [0., 0., 5., 1.]],

                          [[1., 0., 0., 1.],
                           [0., 3., 0., 1.],
                           [0., 0., 5., 1.]]
                          ])
    y_pred = tf.Variable([[[1., 0., 0., 1.],
                           [0., 1., 0., 1.],
                           [0., 0., 1., 1.]],

                          [[1., 0., 0., 1.],
                           [0., 1., 0., 1.],
                           [0., 0., 1., 1.]]
                          ])

    l = CorsNetLoss1()
    print(l(y_true, y_pred))
    print(tf.equal(l(y_true, y_pred), tf.constant(0.375)))
