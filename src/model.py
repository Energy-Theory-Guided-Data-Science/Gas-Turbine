from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from src.loss_batch_history import LossBatchHistory
from src.loss_functions import LossTwoState, LossTwoState2, WeightedLossTwoState, LossTwoStateDiffRange, LossRange, LossMseDiff, LossDiffRange
from src.loss_metrics import MetricLossTwoState, MetricWeightedLossTwoState, MetricLossTwoStateDiffRange, MetricLossRange, MetricLossMseDiff, MetricLossDiffRange
import pandas as pd
from src.utils import create_prediction_plot, create_results_folder
import sklearn.metrics
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Model:
    """
    Class for creating, training and evaluation of the model.
    """

    def __init__(self,
                 data_type,
                 loss_function,
                 scaler,
                 steepness,
                 input_shape,
                 neurons=32,
                 learning_rate=0.001,
                 theta=0.9996,
                 batch_size=128,
                 gpu=0,
                 verbose=True,
                 seed=42,
                 results_folder=None,
                 dropout_rate=0.25,
                 n_lstm_layers=3,
                 ex_name="Test"):
        """
        Initialize the model.
        :param data_type: The type of data used for the model (e.g. "experiment" or "synthetic").
        :param loss_function: The loss function used for the model (e.g. "mean_squared_error" or "custom").
        The custom loss function consists of the mean squared error and the weighted physical part.
        :param scaler: The scaler used for the data.
        :param steepness: The steepness domain knowledge of the loss function. Only used if loss_function is "custom".
        :param input_shape: The shape of the input vector.
        :param neurons: The number of neurons in the first hidden layer. Default is 16.
        :param learning_rate: The learning rate used for the model. Default is 0.001.
        :param theta: The lambda value (weight) of the domain knowledge part in the general loss function.
        Only used if loss_function is "custom". Default is 100.
        :param batch_size: The batch size used for the model. Default is 8.
        :param gpu: The GPU ID used for the model. If set to -1, the CPU is used.
        :param verbose: If True, the training process is printed to the console.
        :param seed: The seed used for the model. Default is 42.
        :param ex_name: The name of the experiment. Default is "Test".
        """
        # tf.random.set_seed(seed)
        # if gpu != -1:
        #    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[gpu], True)
        #    tf.config.set_visible_devices(tf.config.list_physical_devices("GPU")[gpu], "GPU")
        self.ex_name = ex_name
        self.seed = seed
        self.loss_function = loss_function
        if self.loss_function == "mean_squared_error":
            self.approach = "Data_Baseline"
        else:
            self.approach = "Loss_Function"
        self.dropout_rate = dropout_rate
        self.n_lstm_layers = n_lstm_layers
        self.verbose = verbose
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.theta = theta
        self.steepness = steepness
        self.batch_size = batch_size
        self.results_folder = results_folder
        if results_folder is None:
            self.results_folder = create_results_folder(data_type=data_type, approach=self.approach)
        kernel_reg = regularizers.l2(0.05)

        model = Sequential()
        if self.n_lstm_layers == 1:
            model.add(layers.LSTM(self.neurons, input_shape=input_shape, return_sequences=False))
        else:
            model.add(layers.LSTM(self.neurons, input_shape=input_shape, return_sequences=True))
            model.add(layers.Dropout(self.dropout_rate))
            for i in range(self.n_lstm_layers - 2):
                model.add(layers.LSTM(self.neurons, return_sequences=True))
                model.add(layers.Dropout(self.dropout_rate))
            model.add(layers.LSTM(self.neurons, return_sequences=False))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(self.neurons // 2, activation="relu"))
        model.add(layers.Dense(1, activation="linear"))

        # opt = optimizers.Adagrad(learning_rate=self.learning_rate, clipnorm=1.0)
        opt = optimizers.Adam(learning_rate=self.learning_rate)

        metrics = [tf.metrics.RootMeanSquaredError(), tf.metrics.MeanSquaredError()]
        if self.loss_function == "mean_squared_error":
            loss_func = "mean_squared_error"
        elif self.loss_function == 'range':
            min_value, max_value = 0, 4000
            values_to_transform = [[min_value], [max_value]]
            scaled_values = scaler[1].transform(values_to_transform)
            scaled_min_value = scaled_values[0][0]
            scaled_max_value = scaled_values[1][0]
            loss_func = LossRange(self.theta, scaled_min_value, scaled_max_value)
            loss_metric = MetricLossRange(self.theta, scaled_min_value, scaled_max_value)
            metrics.append(loss_metric)
        elif self.loss_function == 'diff_range':
            min_value, max_value = -self.steepness, self.steepness
            values_to_transform = [[min_value], [max_value]]
            scaled_values = scaler[1].transform(values_to_transform)
            scaled_min_value = scaled_values[0][0]
            scaled_max_value = scaled_values[1][0]
            loss_func = LossDiffRange(self.theta, scaled_min_value, scaled_max_value)
            loss_metric = MetricLossDiffRange(self.theta, scaled_min_value, scaled_max_value)
            metrics.append(loss_metric)
        elif self.loss_function == 'mse_diff':
            loss_func = LossMseDiff(self.theta)
            loss_metric = MetricLossMseDiff(self.theta)
            metrics.append(loss_metric)
        elif self.loss_function == 'two_state':
            st = scaler[1].transform([[self.steepness]])[0][0]
            loss_func = LossTwoState(self.theta, st)
            loss_metric = MetricLossTwoState(self.theta, st)
            metrics.append(loss_metric)
        elif self.loss_function == 'two_state_2':
            st = scaler[1].transform([[self.steepness]])[0][0]
            loss_func = LossTwoState2(self.theta, st)
            loss_metric = MetricLossTwoState(self.theta, st)
            metrics.append(loss_metric)
        elif self.loss_function == 'weighted_two_state':
            st = scaler[1].transform([[self.steepness]])[0][0]
            loss_func = WeightedLossTwoState(self.theta, st, tgds_ratio)
            loss_metric = MetricWeightedLossTwoState(self.theta, st)
            metrics.append(loss_metric)
        elif self.loss_function == 'two_state_diff_range':
            st = scaler[1].transform([[self.steepness]])[0][0]
            loss_func = LossTwoStateDiffRange(self.theta, st, tgds_ratio)
            loss_metric = MetricLossTwoStateDiffRange(self.theta, st)
            metrics.append(loss_metric)
        model.compile(loss=loss_func, optimizer=opt, metrics=metrics)
        self.model = model

    def train(self, x_train, y_train, epochs, x_val=None, y_val=None, val_frac=0.1, patience=40, early_stopping=False,
              get_history=False):
        """
        Train the model.
        :param x_train: The training data.
        :param y_train: The training labels.
        :param epochs: The number of epochs used for the training.
        :param val_frac: The fraction of the training data used for validation. Default is 0.1.
        :param patience: The number of epochs without improvement after which the training is stopped. Default is 40.
        :param early_stopping: If True, early stopping is used. Default is True.
        """
        self.epochs = epochs
        start_train = datetime.now()
        loss_batch_history = LossBatchHistory()
        es = EarlyStopping(
            monitor="val_loss", mode="min", verbose=self.verbose, patience=patience, restore_best_weights=True
        )
        mc = ModelCheckpoint(
            self.results_folder + "Model/best_model.h5",
            verbose=self.verbose,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
        callbacks = [mc, TerminateOnNaN(), loss_batch_history]
        if early_stopping:
            callbacks.append(es)
        if x_val is None:
            history = self.model.fit(
                x_train,
                y_train,
                validation_split=val_frac,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                shuffle=False,
                callbacks=callbacks,
            )
        else:
            history = self.model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                shuffle=False,
                callbacks=callbacks,
            )

        self.dur_train = datetime.now() - start_train
        self._save_history(history, loss_batch_history)
        self.load_best_model()

        if get_history:
            return history

    def _save_history(self, history, loss_batch_history):
        """
        Save the training history.
        :param history: The training history.
        :param loss_batch_history: The training history over each batch for all epochs.
        """
        history_dict = {}
        history_dict["loss"] = np.array(history.history["loss"])
        history_dict["val_loss"] = np.array(history.history["val_loss"])
        history_dict["rmse"] = np.array(history.history["root_mean_squared_error"])
        history_dict["val_rmse"] = np.array(history.history["val_root_mean_squared_error"])
        history_dict["mse"] = np.array(history.history["mean_squared_error"])
        history_dict["val_mse"] = np.array(history.history["val_mean_squared_error"])
        history_dict["epoch"] = np.arange(1, len(history_dict["loss"]) + 1)

        if self.loss_function == "mean_squared_error":
            loss_batch_history.history.pop("loss_tgds", None)
        self.best_epoch = np.argmin(history_dict["val_loss"]) + 1
        history_df = pd.DataFrame.from_records(history_dict)
        history_df.to_csv(self.results_folder + "History/history.csv", index=False)
        batch_history_df = pd.DataFrame(loss_batch_history.history)
        batch_history_df.to_csv(self.results_folder + "History/batch_history.csv", index=False)

    def load_best_model(self):
        """
        Load the best model.
        """
        self.model = load_model(self.results_folder + "Model/best_model.h5", compile=False)

    def predict(self, dataset, show_plot=True):
        """
        Predict the labels of the dataset. The predictions are saved in the results folder.
        :param dataset: The full dataset to predict.
        :param show_plot: If True, a plot of the predictions is shown. Default is True.
        """
        self._save_config(dataset)

        result_true_pred = pd.DataFrame(columns=["sample", "true", "pred"])

        def create_predictions(dataset_type, data, index):
            results = pd.DataFrame(index=index, columns=["mse", "rmse", "r2", "mae", "maxae", "mape"])
            for i in range(len(data)):
                x_test, _ = dataset.transform_sample(data[i], make_3d=True)
                preds_scaled = self.model.predict(x_test)
                preds = dataset.scaler[1].inverse_transform(preds_scaled)
                preds = np.array([i[0] for i in preds])

                create_prediction_plot(
                    data[i]["time"],
                    data[i]["el_power"],
                    data[i]["time"][-len(preds):],
                    preds,
                    image_folder=self.results_folder + "Images/",
                    title=f"Predictions using {self.approach} on sample {index[i]} with model trained "
                          + f"on {dataset.n_train_samples} {dataset.data_type} sets",
                    sample_name=index[i],
                    dataset_type=dataset_type,
                    show_plot=show_plot,
                )

                result_true_pred.loc[len(result_true_pred)] = [
                    index[i],
                    data[i]["el_power"].to_list()[-len(preds):],
                    preds.tolist(),
                ]

                results_ex = self._measure_metrics(data[i]["el_power"][-len(preds):], preds)
                results.at[index[i], "mse"] = results_ex["MSE"]
                results.at[index[i], "rmse"] = results_ex["RMSE"]
                results.at[index[i], "r2"] = results_ex["R2"]
                results.at[index[i], "mae"] = results_ex["MAE"]
                results.at[index[i], "maxae"] = results_ex["MaxAE"]
                results.at[index[i], "mape"] = results_ex["MAPE"]

            results = results.reset_index().rename(columns={'index': 'sample_name'})
            results.to_csv(self.results_folder + "results_" + dataset_type + ".csv", index=False)

        start_predict = datetime.now()

        data_test, names_test = [], []
        if dataset.data_type == "synthetic":
            data_test = dataset.data[-1:]
            names_test = dataset.data_names[-1:]
        elif dataset.data_type == "experiment":
            data_test = [dataset.data[i] for i in dataset.test_indices]
            names_test = dataset.test_samples

        create_predictions("test", data_test, names_test)
        create_predictions(
            "train",
            [dataset.data[i] for i in dataset.train_indices],
            [dataset.data_names[i] for i in dataset.train_indices],
        )

        result_true_pred.to_csv(self.results_folder + "results_true_pred.csv", index=False)

        self.dur_predict = datetime.now() - start_predict

    @staticmethod
    def _measure_metrics(values, approx):
        """
        Measure the metrics of the predictions.
        :param values: The true values.
        :param approx: The predicted values.
        :return: A dictionary with the metrics results.
        """
        results = {}
        ms = sklearn.metrics.mean_squared_error(values, approx, squared=True)
        results["MSE"] = ms
        rms = sklearn.metrics.mean_squared_error(values, approx, squared=False)
        results["RMSE"] = rms
        r2 = sklearn.metrics.r2_score(values, approx)
        results["R2"] = r2
        mae = sklearn.metrics.mean_absolute_error(values, approx)
        results["MAE"] = mae
        mape = sklearn.metrics.mean_absolute_percentage_error(values, approx)
        results["MAPE"] = mape
        maxae = sklearn.metrics.max_error(values, approx)
        results["MaxAE"] = maxae
        return results

    def get_result(self, print_result=False):
        """
        Get the results of the prediction. The results are saved in the results folder and printed to the console.
        """

        def delete_best_worst(arr):
            """
            Delete the 5% best and worst values of the array.
            :param arr: The array to delete the values from.
            :return: The array without the 5% best and worst values.
            """
            arr_sorted = sorted(arr)
            return arr_sorted[round(len(arr_sorted) * 0.05): round(len(arr_sorted) * 0.95)]

        results_train = pd.read_csv(self.results_folder + "results_train.csv")
        train_mean = round(np.mean(delete_best_worst(results_train["rmse"])), 3)
        train_std = round(np.std(delete_best_worst(results_train["rmse"])), 3)
        experiment_result = "RMSE (over all train samples): " + str(train_mean) + " ±(" + str(train_std) + ")"

        results_test = pd.read_csv(self.results_folder + "results_test.csv")
        test_mean = round(np.mean(delete_best_worst(results_test["rmse"])), 3)
        test_std = round(np.std(delete_best_worst(results_test["rmse"])), 3)
        experiment_result += "\nRMSE (over all test samples): " + str(test_mean) + " ±(" + str(test_std) + ")"

        # experiment_result += "\nBest Epoch: " + str(self.best_epoch)
        # experiment_result += "\nTraining Time: " + str(self.dur_train)
        # experiment_result += "\nPrediction Time: " + str(self.dur_predict)

        if print_result:
            print(experiment_result)
        self._save_result()

    def _save_result(self):
        """
        Save the results of the experiment in a json file.
        """
        result = {
            # "best_epoch": int(self.best_epoch),
            # "train_time": str(self.dur_train),
            # "predict_time": str(self.dur_predict),
            "train_rmse_mean": round(np.mean(pd.read_csv(self.results_folder + "results_train.csv")["rmse"]), 3),
            "train_rmse_std": round(np.std(pd.read_csv(self.results_folder + "results_train.csv")["rmse"]), 3),
            "test_rmse_mean": round(np.mean(pd.read_csv(self.results_folder + "results_test.csv")["rmse"]), 3),
            "test_rmse_std": round(np.std(pd.read_csv(self.results_folder + "results_test.csv")["rmse"]), 3),
        }
        with open(self.results_folder + "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def _save_config(self, dataset):
        """
        Save the configuration of the experiment in a json file.
        :param dataset: The dataset used for the experiment.
        """
        if os.path.isfile(self.results_folder + "config.json"):
            return False
        config = {
            "ex_name": self.ex_name,
            "approach": self.approach,
            "seed": self.seed,
            "loss_function": self.loss_function,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "neurons": self.neurons,
            "lambda": self.theta,
            "steepness_loss": self.steepness,
            "data_type": dataset.data_type,
            "steepness": dataset.steepness,
            "n_train_samples": dataset.n_train_samples,
            "train_samples": [dataset.data_names[i] for i in dataset.train_indices],
            "lookback": dataset.lag,
            "test_samples": dataset.test_samples
        }
        with open(self.results_folder + "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
