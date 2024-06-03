import tensorflow as tf


class MetricSoftWeightedLossTwoState(tf.keras.metrics.Metric):
    """
    Class for calculation metric based on soft weighted loss.
    :param theta: Weight of the second part of the loss function.
    :param steepness: Expected steepness of the true values.
    :param tgds_ratio: ratio between the static parts and transitions.
    :param tolerance: Tolerance of the expected states.
    """

    def __init__(self, theta, steepness, tgds_ratio=1, tolerance=0, name="loss_tgds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.theta = theta
        self.steepness = steepness
        self.tgds_ratio = tgds_ratio
        self.tolerance = tolerance
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_pred, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: tf.abs(y_pred[1:] - y_shifted[1:]),
                               lambda: tf.constant([0.0], dtype=tf.float32))

        lower_bound_steepness = self.steepness * (1 - self.tolerance)
        upper_bound_steepness = self.steepness * (1 + self.tolerance)
        out_of_range_low = tf.cast(y_difference < lower_bound_steepness, dtype=tf.float32) * (
                y_difference - lower_bound_steepness)
        out_of_range_high = tf.cast(y_difference > upper_bound_steepness, dtype=tf.float32) * (
                y_difference - upper_bound_steepness)
        tgds_error_trans = tf.math.squared_difference(out_of_range_low + out_of_range_high,
                                                      tf.zeros_like(y_difference))
        tgds_squared_sum_trans = tf.reduce_sum(tgds_error_trans)

        upper_bound_zero = self.tolerance * self.steepness
        out_of_range_high = tf.cast(y_difference > upper_bound_zero, dtype=tf.float32) * (
                y_difference - upper_bound_zero)
        tgds_error_static = tf.math.squared_difference(out_of_range_high, tf.zeros_like(y_difference))
        tgds_squared_sum_static = tf.reduce_sum(tgds_error_static)

        # alpha/beta = static/transition
        beta = 1 / (self.tgds_ratio + 1)
        alpha = 1 - beta

        tgds_squared_sum = tf.minimum(alpha * tgds_squared_sum_trans, beta * tgds_squared_sum_static)
        count = tf.cast(tf.shape(y_difference)[0], dtype=tf.float32)

        self.total.assign_add(tgds_squared_sum)
        self.count.assign_add(count)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count) * self.theta

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)
