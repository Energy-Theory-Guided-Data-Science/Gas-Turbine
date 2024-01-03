from glob import glob
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from math import ceil
from src.utils import create_sample_plot
import os
import urllib.request
from zipfile import ZipFile


class Dataset:
    """
    Class for data preprocessing.
    """

    def __init__(self, data_type, n_train_samples, seed=42, steepness=None, test_samples=None, train_pool=None):
        """
        Initialize the dataset.
        :param data_type: The type of data to use. Can be "synthetic" or "experiment".
        :param n_train_samples: The number of samples to use for training.
        :param seed: The seed to use for the random number generator. Default is 42.
        :param steepness: The steepness of the transition part of the synthetic data.
        """
        self.data_type = data_type
        self.seed = seed
        self.train_pool = train_pool
        if self.data_type == "experiment":
            self.steepness = 6.388
        elif steepness is None:
            raise ValueError("Steepness must be specified for synthetic data")
        else:
            self.steepness = steepness
        if test_samples is None:
            if self.data_type == "synthetic":
                self.test_samples = [str(i) for i in range(100, 300)]
            elif self.data_type == "experiment":
                self.test_samples = ["4", "22"]
        else:
            self.test_samples = test_samples
        self.n_train_samples = n_train_samples
        self.train_frame = None
        if self.data_type == "synthetic":
            self._get_synthetic_data()
        elif self.data_type == "experiment":
            self._get_experiment_data()

    def _get_synthetic_data(self):
        """
        Get the synthetic data and split it into train and test sets. If the data is not present, it will be created and saved.
        """
        folder = "Data/Synthetic_Data/quadratic_40_0_linear_" + str(self.steepness) + "/"
        if not glob(folder):
            self._synthesize_data(folder)
        files = sorted(glob(f"{folder}*.csv"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.data = [pd.read_csv(file, delimiter="|") for file in files]
        self.data_names = [str(x) for x in range(len(self.data))]
        self._get_scaler(folder)
        self._get_lag(folder)
        np.random.seed(self.seed)
        self.train_indices = np.random.choice(range(100), self.n_train_samples, replace=False)
        self.train_samples = [self.data_names[i] for i in self.train_indices]
        self.test_indices = [index for index, value in enumerate(self.data_names) if value in self.test_samples]

    def _get_experiment_data(self):
        """
        Get the experiment data and split it into train and test sets. If the data is not present, it will be downloaded and saved.
        """
        folder = "Data/Experiment_Data/"
        if not glob(folder):
            self._download_experiment_data(folder)
        self.data_names = ["1", "4", "9", "20", "21", "22", "23", "24"]
        if self.train_pool is None:
            self.train_pool = self.data_names.copy()
        self.data = [pd.read_csv(f"{folder}experiment_{ex}_short.csv", delimiter="|") for ex in self.data_names]
        # remove unimportant columns
        for i, data in enumerate(self.data):
            if "spinning_soll" in data.columns:
                data = data.drop(["spinning_soll", "th_power"], axis=1)
            else:
                data = data.drop(["spinning_ist", "th_power"], axis=1)
            self.data[i] = data
        self._get_scaler(folder)
        self._get_lag(folder)
        indices = [index for index, value in enumerate(self.data_names) if
                   value not in self.test_samples and value in self.train_pool]
        np.random.seed(self.seed)
        self.train_indices = np.random.choice(indices, self.n_train_samples, replace=False)
        self.train_samples = [self.data_names[i] for i in self.train_indices]
        self.test_indices = [index for index, value in enumerate(self.data_names) if value in self.test_samples]

    def _get_scaler(self, folder):
        """
        Get the scaler for the data.
        :param folder: The folder where the data is stored.
        """
        with open(f"{folder}info.json") as f:
            data_info = json.load(f)
        arr_input = np.array(data_info["input_range"], dtype="float64")
        arr_output = np.array(data_info["output_range"], dtype="float64")
        scaler_input = self._scale(arr_input)[1]
        scaler_output = self._scale(arr_output)[1]
        self.scaler = [scaler_input, scaler_output]

    def _get_lag(self, folder):
        """
        Get the max lag (lookback period length) for the data.
        :param folder: The folder where the data is stored.
        """
        with open(f"{folder}info.json") as f:
            data_info = json.load(f)
        self.lag = data_info["max_lag"]

    @staticmethod
    def _scale(data, scaler=None):
        """
        Scale the data.
        :param data: The data to scale.
        :param scaler: The scaler to use. If None, a new scaler will be created.
        :return: The scaled data and the scaler.
        """
        data = data.reshape(-1, 1)
        if scaler is None:
            scaler = MinMaxScaler()
            scaler = scaler.fit(data)
        data_scaled = scaler.transform(data).reshape(-1)
        return data_scaled, scaler

    # convert series to supervised learning
    @staticmethod
    def _series_to_supervised(data, n_in=1, n_out=1):
        """
        Convert a time series to a supervised learning problem.
        :param data: The time series to convert.
        :param n_in: The number of lag observations as input (X).
        :param n_out: The number of observations as output (y).
        :return: The input and output dataframe.
        """
        n_vars = data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
            else:
                names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg

    def transform_sample(self, sample, make_3d=False):
        """
        Transform a sample to the correct format for the model.
        :param sample: The sample to transform.
        :param make_3d: If True, the sample will be transformed to a 3D array.
        :return: The transformed sample.
        """
        input_scaled = self._scale(np.array(sample["input_voltage"]), scaler=self.scaler[0])[0]
        output_scaled = self._scale(np.array(sample["el_power"]), scaler=self.scaler[1])[0]
        dataset = pd.DataFrame({"input_voltage": input_scaled, "el_power": output_scaled})
        values = dataset.values
        values = values.astype("float64")
        reframed = self._series_to_supervised(values, n_in=self.lag - 1)
        variable_names = list(dataset.keys())
        reframed.columns = list(
            map(
                lambda column: variable_names[int(column[3: column.find("(")]) - 1] + column[column.find("("):],
                list(reframed.columns),
            )
        )
        reframed.set_index(pd.Index(sample["time"][self.lag - 1:]), inplace=True)
        # the current power is a target variable
        y_transformed = reframed[["el_power(t)"]].copy()
        x_transformed = reframed.filter(regex="input_voltage(t*)").copy()
        if make_3d:
            # reshape input to be 3D [samples, timestamps, features]
            n_features = 1
            x_transformed = x_transformed.values
            x_transformed = x_transformed.reshape(x_transformed.shape[0], x_transformed.shape[1], n_features)
            y_transformed = y_transformed.values
        return x_transformed, y_transformed

    def prepare_data(self, indices, lag=None):
        """
        Create the training data.
        :param lag: The lag to use. If None, the lag from the data will be used.
        """
        if lag is not None:
            self.lag = lag

        x, y = pd.DataFrame(), pd.DataFrame()
        train_samples = [self.data[i] for i in indices]
        for sample in train_samples:
            x_ex, y_ex = self.transform_sample(sample)
            x = pd.concat([x, x_ex], ignore_index=True)
            y = pd.concat([y, y_ex], ignore_index=True)
        if self.train_frame is None:
            self.train_frame = pd.concat([x, y], axis=1)
        # reshape input to be 3D [samples, timestamps, features]
        n_features = 1
        x = x.values
        X_train = x.reshape(x.shape[0], x.shape[1], n_features)
        y_train = y.values
        return X_train, y_train

    def _synthesize_data(self, folder):
        """
        Synthesize the data.
        :param folder: The folder where the data is stored.
        """
        os.makedirs(folder)
        stat_parameters = [40, 0, 0]
        stat_function = lambda x, params: params[0] * x ** 2 + params[1] * x + params[2]
        trans_parameters = [self.steepness, 0]
        trans_function = lambda x, params: params[0] * x + params[1]

        def create_synthetic_data(in_values, time):
            """
            Create electric power values.
            :param in_values: The input voltage values.
            :param time: The time values.
            :return: The electric power values.
            """
            output = np.zeros(time[-1])
            stat_vals = np.zeros(len(in_values))

            # use static function to model static values
            for i in range(len(in_values)):
                stat_vals[i] = stat_function(in_values[i], stat_parameters)

            # in the first time sequence we exclude the transition
            for j in range(time[0], time[1]):
                output[j] = stat_vals[0]

            for t in range(len(time) - 1):
                for i in range(time[t], time[t + 1]):
                    # if rise in input voltage
                    if in_values[t] > in_values[t - 1]:
                        if output[i - 1] <= stat_vals[t]:
                            trans_value = output[time[t] - 1] + trans_function(i - time[t], trans_parameters)
                            value = min(trans_value, stat_vals[t])
                        else:
                            trans_value = output[time[t] - 1] - trans_function(i - time[t], trans_parameters)
                            value = max(trans_value, stat_vals[t])
                        output[i] = value

                    # if drop in input voltage
                    else:
                        if output[i - 1] >= stat_vals[t]:
                            trans_value = output[time[t] - 1] - trans_function(i - time[t], trans_parameters)
                            value = max(trans_value, stat_vals[t])
                        else:
                            trans_value = output[time[t] - 1] + trans_function(i - time[t], trans_parameters)
                            value = min(trans_value, stat_vals[t])
                        output[i] = value
            return output

        def fill_data(time_splits, value_splits, df_time):
            """
            Interpolate input voltage values over time.
            :param time_splits: The time splits.
            :param value_splits: The input voltage values.
            :param df_time: The time values.
            :return: The interpolated input voltage values.
            """
            values = np.empty(len(df_time))

            for i in range(len(time_splits) - 1):
                lower = ceil(time_splits[i])  # find beginning of time frame
                upper = ceil(time_splits[i + 1])  # find end of time frame

                ix_start = np.argmax(df_time >= lower)  # find index where lower timestep is surpassed
                ix_end = np.argmax(df_time > upper)  # find index where lower timestep is surpassed

                v = value_splits[i]
                values[ix_start:ix_end] = v
            values[time_splits[-2]: time_splits[-1]] = value_splits[-2]
            values[time_splits[-1]:] = value_splits[-1]
            return values

        def get_statistic_info():
            """
            Get the statistic information.
            :return: The maximum static, transitions and general periods lengths.
            """
            files = sorted(glob(f"{folder}*.csv"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            synthetic_data = [pd.read_csv(file, delimiter="|") for file in files]
            static_sizes = []
            slope_sizes = []
            lengths = []

            for data in synthetic_data:
                df = data.copy()
                df.drop(columns=["time", "input_voltage"], inplace=True)
                df["el_power"] = df["el_power"].diff(-1).dropna()
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                df_agg = (
                    df.groupby((df["el_power"] != df["el_power"].shift()).cumsum()).agg(list).reset_index(drop=True)
                )
                df_agg["is_static"] = df_agg["el_power"].apply(lambda x: max(x) == 0 and min(x) == 0)
                df_agg = (
                    df_agg.groupby((df_agg["is_static"] != df_agg["is_static"].shift()).cumsum())
                    .agg(list)
                    .reset_index(drop=True)
                )
                df_agg["is_static"] = df_agg["is_static"].apply(lambda x: sum(x) == 1)
                df_agg["el_power"] = df_agg["el_power"].apply(lambda x: [j for sub in x for j in sub])
                df_agg["len"] = df_agg["el_power"].apply(len)

                static_sizes += df_agg[df_agg["is_static"]]["len"].to_list()
                slope_sizes += df_agg[~df_agg["is_static"]]["len"].to_list()
                lengths.append(len(data))
            return max(static_sizes), max(slope_sizes), max(lengths)

        print("Synthesising data...")
        np.random.seed(123)
        for i in tqdm(range(300)):
            jumps = np.random.randint(2, 7)
            input_values = 7 * np.random.random(2 * jumps) + 3
            input_values = np.insert(input_values, 0, 0)
            time_values = np.random.randint(1000, 10000, size=2 * jumps - 1)
            time_values = np.append(time_values, 0)
            time_values = np.append(time_values, 1000)
            time_values.sort()

            sample = pd.DataFrame()
            sample["el_power"] = create_synthetic_data(input_values, time_values)
            sample["time"] = range(time_values[-1])
            sample["input_voltage"] = fill_data(time_values, input_values, sample["time"])
            sample.to_csv(folder + f"sample_{i}.csv", index=False, sep="|")

            create_sample_plot(
                sample, show_plot=False, ylim=stat_function(10.5, stat_parameters), folder=folder, title=f"sample_{i}"
            )

        input_limits = [-1, 10]
        output_limits = [-1, stat_function(input_limits[1], stat_parameters)]
        max_static_size, max_slope_size, max_length = get_statistic_info()
        info = {
            "input_range": input_limits,
            "output_range": output_limits,
            "max_static_size": max_static_size,
            "max_lag": max_slope_size,
            "max_length": max_length,
        }
        with open(folder + "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

    @staticmethod
    def _download_experiment_data(folder):
        """
        Download the experiment data.
        :param folder: The folder to save the data in.
        """
        print("Downloading data...")
        os.makedirs(folder, exist_ok=True)
        path = folder[: folder[:-1].rfind("/") + 1]
        url = "https://bwsyncandshare.kit.edu/s/j7ExGsHaN2wFgWa/download"
        file_name = "data.zip"
        _ = urllib.request.urlretrieve(url, path + file_name)
        with ZipFile(path + file_name, "r") as f:
            f.extractall(path=path)
        os.remove(path + file_name)
        print("Data downloaded.")
