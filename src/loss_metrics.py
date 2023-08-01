import tensorflow as tf


class MetricLossTwoState(tf.keras.metrics.Metric):
    def __init__(self, theta, steepness, name="loss_tgds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.theta = theta
        self.steepness = steepness
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_pred, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: tf.abs(y_pred[1:] - y_shifted[1:]),
                               lambda: tf.constant([0.0], dtype=tf.float32))

        tgds_error_trans = tf.math.squared_difference(y_difference, tf.ones_like(y_difference) * self.steepness)
        tgds_squared_sum_trans = tf.reduce_sum(tgds_error_trans)
        tgds_error_static = tf.math.squared_difference(y_difference, tf.zeros_like(y_difference))
        tgds_squared_sum_static = tf.reduce_sum(tgds_error_static)

        tgds_squared_sum = tf.minimum(tgds_squared_sum_trans, tgds_squared_sum_static)
        count = tf.cast(tf.shape(y_difference)[0], dtype=tf.float32)

        self.total.assign_add(tgds_squared_sum)
        self.count.assign_add(count)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count) * self.theta

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


class MetricWeightedLossTwoState(tf.keras.metrics.Metric):
    def __init__(self, theta, steepness, tgds_ratio=1, name="loss_tgds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.theta = theta
        self.steepness = steepness
        self.tgds_ratio = tgds_ratio
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_pred, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: tf.abs(y_pred[1:] - y_shifted[1:]),
                               lambda: tf.constant([0.0], dtype=tf.float32))

        tgds_error_trans = tf.math.squared_difference(y_difference, tf.ones_like(y_difference) * self.steepness)
        tgds_squared_sum_trans = tf.reduce_sum(tgds_error_trans)
        tgds_error_static = tf.math.squared_difference(y_difference, tf.zeros_like(y_difference))
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


class MetricLossTwoStateDiffRange(tf.keras.metrics.Metric):
    def __init__(self, theta, min_value, max_value, tgds_ratio=1, name="loss_tgds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.theta = theta
        self.min_value = min_value
        self.max_value = max_value
        self.tgds_ratio = tgds_ratio
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_pred, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: tf.abs(y_pred[1:] - y_shifted[1:]),
                               lambda: tf.constant([0.0], dtype=tf.float32))

        out_of_range_diff_low = tf.cast(y_difference < self.min_value, dtype=tf.float32) * (
                y_difference - self.min_value)
        out_of_range_diff_high = tf.cast(y_difference > self.max_value, dtype=tf.float32) * (
                y_difference - self.max_value)

        tgds_error_trans = tf.math.squared_difference(out_of_range_diff_low + out_of_range_diff_high,
                                                      tf.zeros_like(y_difference))
        tgds_squared_sum_trans = tf.reduce_sum(tgds_error_trans)

        tgds_error_static = tf.math.squared_difference(y_difference, tf.zeros_like(y_difference))
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


class MetricLossMseDiff(tf.keras.metrics.Metric):
    def __init__(self, theta, name="loss_tgds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.theta = theta
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_true_shifted = tf.roll(y_true, shift=1, axis=0)
        y_pred_shifted = tf.roll(y_pred, shift=1, axis=0)

        y_true_difference = tf.cond(tf.size(y_true) > 1,
                                    lambda: tf.abs(y_true[1:] - y_true_shifted[1:]),
                                    lambda: tf.constant([0.0], dtype=tf.float32))
        y_pred_difference = tf.cond(tf.size(y_pred) > 1,
                                    lambda: tf.abs(y_pred[1:] - y_pred_shifted[1:]),
                                    lambda: tf.constant([0.0], dtype=tf.float32))

        tgds_error = tf.math.squared_difference(y_true_difference, y_pred_difference)
        tgds_squared_sum = tf.reduce_sum(tgds_error)
        count = tf.cast(tf.shape(y_true_difference)[0], dtype=tf.float32)

        self.total.assign_add(tgds_squared_sum)
        self.count.assign_add(count)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count) * self.theta

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


class MetricLossRange(tf.keras.metrics.Metric):
    def __init__(self, theta, min_value, max_value, name="loss_tgds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.theta = theta
        self.min_value = min_value
        self.max_value = max_value
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        out_of_range_low = tf.cast(y_pred < self.min_value, dtype=tf.float32) * (y_pred - self.min_value)
        out_of_range_high = tf.cast(y_pred > self.max_value, dtype=tf.float32) * (y_pred - self.max_value)

        tgds_error = tf.math.squared_difference(out_of_range_low + out_of_range_high, tf.zeros_like(y_pred))
        tgds_squared_sum = tf.reduce_sum(tgds_error)
        count = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        self.total.assign_add(tgds_squared_sum)
        self.count.assign_add(count)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count) * self.theta

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


class MetricLossDiffRange(tf.keras.metrics.Metric):
    def __init__(self, theta, min_value, max_value, name="loss_tgds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.theta = theta
        self.min_value = min_value
        self.max_value = max_value
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_pred, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: y_pred[1:] - y_shifted[1:],
                               lambda: tf.constant([0.0], dtype=tf.float32))

        out_of_range_diff_low = tf.cast(y_difference < self.min_value, dtype=tf.float32) * (
                y_difference - self.min_value)
        out_of_range_diff_high = tf.cast(y_difference > self.max_value, dtype=tf.float32) * (
                y_difference - self.max_value)

        tgds_error = tf.math.squared_difference(out_of_range_diff_low + out_of_range_diff_high,
                                                tf.zeros_like(y_difference))
        tgds_squared_sum = tf.reduce_sum(tgds_error)
        count = tf.cast(tf.shape(y_difference)[0], dtype=tf.float32)

        self.total.assign_add(tgds_squared_sum)
        self.count.assign_add(count)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count) * self.theta

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)
