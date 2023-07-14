import tensorflow as tf


class LossTwoState(tf.keras.losses.Loss):
    """
        The loss function consists of two parts. The first part is the mean squared error between the true and the predicted values.
        The second part is the mean squared error between expected domain knowledge parameters and the absolute differences
        between the differences of the true values and the differences of the predicted values.
        :param theta: Weight of the second part of the loss function.
        :param steepness: Expected steepness of the true values.
    """

    def __init__(self, theta, steepness, name="loss_two_state"):
        super().__init__(name=name)
        self.theta = theta
        self.steepness = steepness

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_predicted = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_predicted, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: tf.abs(y_predicted[1:] - y_shifted[1:]),
                               lambda: tf.constant([0.0], dtype=tf.float32))

        criterion = tf.keras.losses.MeanSquaredError()

        loss1 = criterion(y_true, y_predicted)

        tgds_loss_trans = criterion(y_difference, tf.ones_like(y_difference) * self.steepness)
        tgds_loss_static = criterion(y_difference, tf.zeros_like(y_difference))

        tgds_loss = tf.minimum(tgds_loss_static, tgds_loss_trans)

        loss = loss1 + self.theta * tgds_loss

        return loss


class LossMseDiff(tf.keras.losses.Loss):
    def __init__(self, theta, name="loss_mse_diff"):
        super().__init__(name=name)
        self.theta = theta

    def call(self, y_true, y_pred):
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

        criterion = tf.keras.losses.MeanSquaredError()

        loss1 = criterion(y_true, y_pred)

        tgds_loss = criterion(y_true_difference, y_pred_difference)

        loss = loss1 + self.theta * tgds_loss

        return loss


class LossRange(tf.keras.losses.Loss):
    def __init__(self, theta, min_value, max_value, name="loss_range"):
        super().__init__(name=name)
        self.theta = theta
        self.min_value = min_value
        self.max_value = max_value

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_predicted = tf.reshape(y_pred, [-1])

        criterion = tf.keras.losses.MeanSquaredError()
        loss1 = criterion(y_true, y_predicted)

        # Calculate additional loss for out-of-range predictions
        out_of_range_low = tf.cast(y_predicted < self.min_value, dtype=tf.float32) * (y_predicted - self.min_value)
        out_of_range_high = tf.cast(y_predicted > self.max_value, dtype=tf.float32) * (y_predicted - self.max_value)
        range_loss = criterion(out_of_range_low + out_of_range_high, tf.zeros_like(y_predicted))

        loss = loss1 + self.theta * range_loss

        return loss


class LossDiffRange(tf.keras.losses.Loss):
    def __init__(self, theta, min_value, max_value, name="loss_diff_range"):
        super().__init__(name=name)
        self.theta = theta
        self.min_value = min_value
        self.max_value = max_value

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_predicted = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_predicted, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: y_predicted[1:] - y_shifted[1:],
                               lambda: tf.constant([0.0], dtype=tf.float32))

        criterion = tf.keras.losses.MeanSquaredError()
        loss1 = criterion(y_true, y_predicted)

        # Calculate additional loss for out-of-range differences
        out_of_range_diff_low = tf.cast(y_difference < self.min_value, dtype=tf.float32) * (
                y_difference - self.min_value)
        out_of_range_diff_high = tf.cast(y_difference > self.max_value, dtype=tf.float32) * (
                y_difference - self.max_value)
        diff_loss = criterion(out_of_range_diff_low + out_of_range_diff_high, tf.zeros_like(y_difference))

        loss = loss1 + self.theta * diff_loss

        return loss
