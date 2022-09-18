import tensorflow as tf
import numpy as np


def get(name):
    """
    Returns the loss function with the given name.
    :param name: Name of the loss function.
    :return: Loss function.
    """
    if name == "custom":
        return loss_function
    if name == "custom_relu":
        return loss_function_relu


def loss_function(theta, steepness, batch_size):
    """
    The loss function consists of two parts. The first part is the mean squared error between the true and the predicted values.
    The second part is the mean squared error between expected domain knowledge parameters and the absolute differences
    between the differences of the true values and the differences of the predicted values.
    :param theta: Weight of the second part of the loss function.
    :param steepness: Expected steepness of the true values.
    :param batch_size: Size of the batch.
    """

    def loss_function_diff(y_true, y_predicted):
        y_true = tf.raw_ops.Reshape(tensor=y_true, shape=tf.raw_ops.Shape(input=y_predicted))
        y_true = tf.raw_ops.Reshape(tensor=y_true, shape=[-1])
        y_predicted = tf.raw_ops.Reshape(tensor=y_predicted, shape=[-1])

        y_shifted = tf.raw_ops.Roll(input=y_true, shift=1, axis=0)
        y_difference = tf.raw_ops.Sub(x=y_predicted, y=y_shifted)

        known_diff_trans = np.full(batch_size - 1, steepness)
        known_tensor_trans = tf.convert_to_tensor(known_diff_trans, dtype="float32")
        known_diff_static = np.full(batch_size - 1, 0)
        known_tensor_static = tf.convert_to_tensor(known_diff_static, dtype="float32")

        criterion = tf.keras.losses.MeanSquaredError()

        loss1 = criterion(y_true, y_predicted)

        tgds_loss_trans = criterion(tf.raw_ops.Abs(x=y_difference[1:]), known_tensor_trans[: tf.size(y_difference) - 1])
        tgds_loss_static = criterion(
            tf.raw_ops.Abs(x=y_difference[1:]), known_tensor_static[: tf.size(y_difference) - 1]
        )

        tgds_loss = tf.raw_ops.Minimum(x=tgds_loss_static, y=tgds_loss_trans)

        for_test = tf.constant([0.0], dtype=tf.float32)
        tgds_loss = tf.where(tf.math.greater(tf.size(y_difference), tf.size(for_test)), tgds_loss, for_test)

        loss = tf.raw_ops.Add(x=loss1, y=tf.math.scalar_mul(theta, tgds_loss))

        return loss

    return loss_function_diff


def loss_function_relu(theta, steepness, batch_size):
    """
    The ReLU implementation of the loss function.
    :param theta: Weight of the second part of the loss function.
    :param steepness: Expected steepness of the true values.
    :param batch_size: Size of the batch.
    """

    def loss_function_diff(y_true, y_predicted):
        y_true = tf.raw_ops.Reshape(tensor=y_true, shape=tf.raw_ops.Shape(input=y_predicted))
        y_true = tf.raw_ops.Reshape(tensor=y_true, shape=[-1])
        y_predicted = tf.raw_ops.Reshape(tensor=y_predicted, shape=[-1])

        y_shifted = tf.raw_ops.Roll(input=y_true, shift=1, axis=0)
        y_difference = tf.raw_ops.Sub(x=y_predicted, y=y_shifted)

        known_diff = np.full(batch_size - 1, steepness)
        known_tensor = tf.convert_to_tensor(known_diff, dtype="float32")

        criterion = tf.keras.losses.MeanSquaredError()

        loss1 = criterion(y_true, y_predicted)

        tgds_loss = tf.raw_ops.Sub(x=tf.raw_ops.Abs(x=y_difference[1:]), y=known_tensor[: tf.size(y_difference) - 1])
        tgds_loss = tf.keras.activations.relu(tgds_loss)

        test_size = tf.constant([0.0], dtype=tf.float32)
        tgds_loss = tf.where(
            tf.raw_ops.GreaterEqual(x=tf.size(y_difference), y=tf.size(test_size)), tgds_loss, test_size
        )

        tgds_loss = tf.math.reduce_sum(tgds_loss)

        loss = tf.raw_ops.Add(x=loss1, y=tf.math.scalar_mul(theta, tgds_loss))

        return loss

    return loss_function_diff
