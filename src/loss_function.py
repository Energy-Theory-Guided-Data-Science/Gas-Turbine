import tensorflow as tf


class SoftWeightedLossTwoState(tf.keras.losses.Loss):
    """
        The loss function consists of two parts. The first part is the mean squared error between the true and the predicted values.
        The second part is the mean squared error between expected domain knowledge parameters and the absolute differences
        between the differences of the true values and the differences of the predicted values.
        :param theta: Weight of the second part of the loss function.
        :param steepness: Expected steepness of the true values.
        :param tgds_ratio: ratio between the static parts and transitions.
        :param tolerance: Tolerance of the expected states.
        :param normalized: If true, the tgds_ratio is normalized to 1 in sum.
    """

    def __init__(self, theta, steepness, tgds_ratio=1, tolerance=0.1, normalized=False,
                 name="soft_loss_weighted_two_state"):
        super().__init__(name=name)
        self.theta = theta
        self.steepness = steepness
        self.tgds_ratio = tgds_ratio
        self.normalized = normalized
        self.tolerance = tolerance

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_predicted = tf.reshape(y_pred, [-1])

        y_shifted = tf.roll(y_predicted, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1,
                               lambda: tf.abs(y_predicted[1:] - y_shifted[1:]),
                               lambda: tf.constant([0.0], dtype=tf.float32))

        criterion = tf.keras.losses.MeanSquaredError()
        loss1 = criterion(y_true, y_predicted)

        # Tolerance bounds for transition
        lower_bound_steepness = self.steepness * (1 - self.tolerance)
        upper_bound_steepness = self.steepness * (1 + self.tolerance)

        # Transition steepness loss
        deviation_from_steepness = tf.where((y_difference < lower_bound_steepness),
                                            lower_bound_steepness - y_difference,
                                            tf.where((y_difference > upper_bound_steepness),
                                                     y_difference - upper_bound_steepness,
                                                     tf.zeros_like(y_difference)))

        tgds_loss_trans = criterion(deviation_from_steepness, tf.zeros_like(deviation_from_steepness))

        # Static loss calculation with tolerance
        upper_bound_zero = self.tolerance * self.steepness
        deviation_from_zero = tf.where(y_difference > upper_bound_zero,
                                       y_difference - upper_bound_zero,
                                       tf.zeros_like(y_difference))

        tgds_loss_static = criterion(deviation_from_zero, tf.zeros_like(deviation_from_zero))

        # Weighted combination of static and transition losses
        beta = 1 / (self.tgds_ratio + 1)
        alpha = 1 - beta
        tgds_loss = tf.minimum(alpha * tgds_loss_static, beta * tgds_loss_trans)

        if self.normalized:
            loss = (1 - self.theta) * loss1 + self.theta * tgds_loss
        else:
            loss = loss1 + self.theta * tgds_loss

        return loss
