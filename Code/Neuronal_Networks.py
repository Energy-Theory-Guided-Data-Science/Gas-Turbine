################################################################################
######################## Import of libraries ###################################
################################################################################
# import own libraries
import Data_Processing as dp
import Global_Functions as gf
# import python libraries
import numpy as np
import pandas as pd
import math
# import tensorflow
from tensorflow.keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping


def generate_batches(X, y, batch_size = 1):
    i = 0
    while True:
        X_batch = []
        y_batch = []
        for b in range(batch_size):
            if i >= len(X):
                i = np.random.randint(len(X))
            X_point = X[i, :]
            y_point = y[i, :]
            i += 1
            
            X_batch.append(X_point)
            y_batch.append(y_point)
        yield (np.array(X_batch), np.array(y_batch))


def fit_lstm(X_train, y_train, X_val, y_val, batch_size, nb_epochs, neurons, loss_function = 'mean_squared_error', out_of_last_layer = 2):
    steps_per_epoch = int(math.floor((1. * len(X_train)) / batch_size))
    val_steps = int(math.floor((1. * len(X_val)) / batch_size))    
    history = list()
    
    model = Sequential()
    model.add(layers.LSTM(int(neurons/2), batch_input_shape = (batch_size, X_train.shape[1], X_train.shape[2]), stateful = True, return_sequences = True, kernel_regularizer = 'l2'))
    model.add(layers.Dense(int(neurons/2), kernel_regularizer = 'l2'))
    model.add(layers.Dense(out_of_last_layer, kernel_regularizer = 'l2'))
    model.compile(loss = loss_function, optimizer = 'adam', )
    
    for i in range(nb_epochs):
        history.append(model.fit(generate_batches(X_train, y_train, batch_size = batch_size), validation_data = generate_batches(X_val, y_val, batch_size = batch_size), validation_steps = val_steps, epochs = 1, steps_per_epoch = steps_per_epoch, verbose = 0, shuffle = False))
        model.reset_states()
        if i % 10 == 9:
            print('Epoch {} of {} is done.'.format(str(i+1), nb_epochs))
    return model, history

# prepare data of train and validation data frame based on the chosen parameters.
# The preparation for this file follows the specification of 'add' for differences. Please refer to the function 'prepare_data' in the file 'Data_Processing' for more information.
def train_model(experiment_train, experiment_validation, difference_chosen = None,
                loss_function = 'mean_squared_error',
                save_folder = "../Models/", all_lags = False, hybrid = False,
                lag_chosen=60, batch_size = 1, nmb_epochs=20, neurons_chosen=64):
    gf.check_folder(save_folder)
    # prepare data with the method in the utility file 'Data Preprocessing'
    scaler_train, X_train, y_train = dp.prepare_data(experiment_train,
                                                     lag = lag_chosen, all_lags = all_lags,
                                                     differences = difference_chosen, hybrid = hybrid)
    scaler_val, X_val, y_val = dp.prepare_data(experiment_validation,
                                                lag = lag_chosen, all_lags = all_lags,
                                               differences = difference_chosen, hybrid = hybrid)
    # train the model using the function above
    model, history = fit_lstm(X_train, y_train, X_val, y_val,
                                 loss_function = loss_function,
                                 batch_size = batch_size,
                                 nb_epochs = nmb_epochs,
                                 neurons = neurons_chosen,
                                 out_of_last_layer = y_train.shape[2])
    model.save(save_folder + 'model.h5')

    return model, history, scaler_train, X_train, y_train, scaler_val, X_val, y_val


# Predict the values given an experiment and a corresponding model
def predictions(experiment, model, difference_chosen,
                all_lags = False, hybrid = False, lag_chosen = 60, batch_size = 1):
    # prepare data with the method in the utility file 'Data Preprocessing'
    scaler, X, y = dp.prepare_data(experiment, lag = lag_chosen, all_lags = all_lags,
                                   differences = difference_chosen, hybrid = hybrid)
    
    y = np.array([i[0][0] for i in y])
    steps = int(math.floor((1. * len(X)) / batch_size))
    preds_scaled = model.predict(X[:steps*batch_size], batch_size = batch_size)
    preds_scaled = np.array([i[0] for i in preds_scaled])
    preds = scaler[1].inverse_transform(preds_scaled) # scaler[1] is the electric power scaler, adapt if neccessary.
    preds = np.array([i[0] for i in preds])
    
    return scaler, X, y, preds_scaled, preds