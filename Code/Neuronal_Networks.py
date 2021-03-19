import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

def scale(data_series, scaler = None):
    data = data_series.values
    data = data.reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler(feature_range = (-1,1), )
        scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler

def prepare_data(data, lag_input = 1, lag_output = 1, all_lags = True, differences = 'none', hybrid = False):
    length = len(data)
    input_scaled, scaler_input = scale(data['input_voltage'])
    power_scaled, scaler_power = scale(data['el_power'])
    
    scaler = [scaler_input, scaler_power]
    
    input_scaled = [x[0] for x in input_scaled]
    power_scaled = [x[0] for x in power_scaled]
    
    input_scaled = np.append(input_scaled, np.full(lag_input, -1))
    power_scaled = np.append(power_scaled, np.full(lag_output, -1))
    
    df = pd.DataFrame()
    
    for i in range(lag_input):
        df['input_voltage_delay_' + str(i)] = np.roll(input_scaled, i)[:length]
    
    for j in range(lag_output):
        df['el_power_delay_' + str(j)] = np.roll(power_scaled, i)[:length]
    
    y = df[['el_power_delay_0']]
    
    if all_lags:
        X = df.filter(regex = ("input_voltage_delay_.*"))
    else:
        X = df[['input_voltage_delay_0', 'input_voltage_delay_' +str(lag_input-1)]]
        
    if differences == 'add':
        X['difference'] = data.diff(lag_input)['el_power']
        X['difference'][:lag_input] = 0
    elif differences == 'only':
        diffs = data.diff(lag_input)['el_power']
        diffs[:lag_input] = 0
        X = pd.DataFrame(diffs)
	
    if hybrid:
        theor_preds_scaled, scaler_theor = scale(data['theor_predictions'])
        scaler.append(scaler_theor)
        X['theor_pred'] = theor_preds_scaled
	
    X = X.values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    return scaler, X, y
	
def fit_lstm(X_train, y_train, X_val, y_val, batch_size, nb_epochs, neurons):    
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
    history = list()

    model = Sequential()
    model.add(layers.LSTM(neurons, batch_input_shape = (batch_size, X_train.shape[1], X_train.shape[2]), stateful = True))
    model.add(layers.Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', )
    for i in range(nb_epochs):
        history.append(model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 1, batch_size = batch_size, verbose = 1, shuffle = False, callbacks = [es]))
        model.reset_states()
    return model, history