import tensorflow as tf
import numpy as np


def get(name):
    """
    Returns the loss function with the given name.
    :param name: Name of the loss function.
    :return: Loss function.
    """
    if name == "two_state":
        return loss_function_two_state
    if name == 'range':
        return loss_func_range
    if name == 'diff_range':
        return loss_func_diff_range
    if name == 'mse_diff':
        return loss_func_mse_diff


def loss_function_two_state(theta, steepness):
    """
    The loss function consists of two parts. The first part is the mean squared error between the true and the predicted values.
    The second part is the mean squared error between expected domain knowledge parameters and the absolute differences
    between the differences of the true values and the differences of the predicted values.
    :param theta: Weight of the second part of the loss function.
    :param steepness: Expected steepness of the true values.
    :param batch_size: Size of the batch.
    """

    def loss_function(y_true, y_predicted):
        y_true = tf.reshape(y_true, [-1])
        y_predicted = tf.reshape(y_predicted, [-1])

        y_shifted = tf.roll(y_predicted, shift=1, axis=0)      
        y_difference = tf.cond(tf.size(y_true) > 1, 
                               lambda: tf.abs(y_predicted[1:] - y_shifted[1:]), 
                               lambda: tf.constant([0.0], dtype=tf.float32))
        
        criterion = tf.keras.losses.MeanSquaredError()

        loss1 = criterion(y_true, y_predicted)

        tgds_loss_trans = criterion(y_difference, tf.ones_like(y_difference) * steepness)
        tgds_loss_static = criterion(y_difference, tf.zeros_like(y_difference))

        tgds_loss = tf.minimum(tgds_loss_static, tgds_loss_trans)

        loss = loss1 + theta * tgds_loss

        return loss

    return loss_function


def loss_func_mse_diff(prm_tgds):
    def loss_function(y_true, y_pred):
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

        loss = loss1 + prm_tgds * tgds_loss

        return loss

    return loss_function

def loss_func_range(prm_tgds, min_value, max_value):
    def loss_function(y_true, y_predicted):
        y_true = tf.reshape(y_true, [-1])
        y_predicted = tf.reshape(y_predicted, [-1])

        criterion = tf.keras.losses.MeanSquaredError()
        loss1 = criterion(y_true, y_predicted)

        # Calculate additional loss for out-of-range predictions
        out_of_range_low = tf.cast(y_predicted < min_value, dtype=tf.float32) * (y_predicted - min_value)
        out_of_range_high = tf.cast(y_predicted > max_value, dtype=tf.float32) * (y_predicted - max_value)
        range_loss = criterion(out_of_range_low + out_of_range_high, tf.zeros_like(y_predicted))

        loss = loss1 + prm_tgds * range_loss

        return loss

    return loss_function

def loss_func_diff_range(prm_tgds, min_diff, max_diff):
    def loss_function(y_true, y_predicted):
        y_true = tf.reshape(y_true, [-1])
        y_predicted = tf.reshape(y_predicted, [-1])

        y_shifted = tf.roll(y_predicted, shift=1, axis=0)
        y_difference = tf.cond(tf.size(y_true) > 1, 
                               lambda: y_predicted[1:] - y_shifted[1:], 
                               lambda: tf.constant([0.0], dtype=tf.float32))
        
        criterion = tf.keras.losses.MeanSquaredError()
        loss1 = criterion(y_true, y_predicted)
        
        # Calculate additional loss for out-of-range differences
        out_of_range_diff_low = tf.cast(y_difference < min_diff, dtype=tf.float32) * (y_difference - min_diff)
        out_of_range_diff_high = tf.cast(y_difference > max_diff, dtype=tf.float32) * (y_difference - max_diff)
        diff_loss = criterion(out_of_range_diff_low + out_of_range_diff_high, tf.zeros_like(y_difference))

        loss = loss1 + prm_tgds * diff_loss

        return loss

    return loss_function
