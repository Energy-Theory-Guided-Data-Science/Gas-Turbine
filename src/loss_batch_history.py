import tensorflow as tf


class LossBatchHistory(tf.keras.callbacks.Callback):
    """
    Class for storing the loss history of a model during training.
    """

    def __init__(self):
        super(LossBatchHistory, self).__init__()
        self.history = {
            "epoch": [],
            "batch": [],
            "loss": [],
            "loss_mse": [],
            "loss_tgds": []
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        """
        Stores the loss of the current batch.
        :param batch: The number of the current batch.
        :param logs: The logs of the training.
        """
        self.history["epoch"].append(self.current_epoch)
        self.history["batch"].append(batch)
        self.history["loss"].append(logs.get("loss"))
        self.history["loss_mse"].append(logs.get("mean_squared_error"))
        try:
            self.history["loss_tgds"].append(logs.get("loss_tgds"))
        except:
            pass
