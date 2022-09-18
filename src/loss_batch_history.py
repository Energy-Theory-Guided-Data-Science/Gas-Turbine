import tensorflow as tf


class LossBatchHistory(tf.keras.callbacks.Callback):
    """
    Class for storing the loss history of a model during training.
    """

    def on_train_begin(self, logs=None):
        """
        Initializes the history dictionary.
        :param logs: The logs of the training.
        """
        self.history = {"loss": [], "mse": [], "physical_loss": []}

    def on_batch_end(self, batch, logs=None):
        """
        Stores the loss of the current batch.
        :param batch: The number of the current batch.
        :param logs: The logs of the training.
        """
        self.history["loss"].append(logs.get("loss"))
        self.history["mse"].append(logs.get("mean_squared_error"))
        try:
            self.history["physical_loss"].append(logs.get("loss_function_diff"))
        except:
            pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Highlights the end of the current epoch.
        :param epoch: The number of the current epoch.
        :param logs: The logs of the training.
        """
        self.history["loss"].append(-1)
        self.history["mse"].append(-1)
        self.history["physical_loss"].append(-1)
