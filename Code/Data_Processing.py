import lttb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
################################################################################
####################### Preperation of data ####################################
################################################################################

# downsample data according to sample size
def downsample_data(time, values, sample_size = 12232):
    assert len(time) == len(values), "The data to be downsampled has to have the same length. Lengths in this case are " + len(time) + " for time and " + len(values) + "for values."
    data = np.array([time, values]).T
    while len(data.shape) != 2:
        data = data[0]
    downsampled_data = lttb.downsample(data, n_out = sample_size)
    assert downsampled_data.shape[0] == sample_size, "The downsampled value does not match the specified size. Somehow my code didn't work, I'm sorry."
    return downsampled_data[:,1]

# scale the data to [-1,1] with a given scaler. If no sclaer is given scale the data accordingly and return the new scaler.
def scale(data_series, scaler = None):
    data = data_series.values
    data = data.reshape(-1, 1)
    if scaler is None:
        scaler = StandardScaler()
        scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled = [x[0] for x in data_scaled]
    return data_scaled, scaler


# prepare the data for the RNNs.
# The parameters do as following:
#      - data: [pd.DataFrame] the data frame containing input and output values
#      - lag: [int] determine the window how far back in time we look for our data
#      - all_lags: [boolean] set True to use all lags, False to use only the current and the last
#      - differences: [string] specify which method should be used for the preparation. Current methods include 'add' (include differences in input --> Intermediate (22)), 'predict' (include in output --> RNN predict Diff (31)) and 'input' (include difference of inputs in model input).
#      - hybrid: [boolean] if True include theoretical predictions in model training.
def prepare_data(data, scaler = None, lag = 60, all_lags = True,
                 differences = None, hybrid = False):
    df = pd.DataFrame()
    
    # first scale the given data and put the scalers in a list
    if scaler is None:
        scaler_in = None
        scaler_pow = None
    else:
        scaler_in = scaler[0]
        scaler_pow = scaler[1]
    input_scaled, scaler_input = scale(data['input_voltage'], scaler = scaler_in)
    power_scaled, scaler_power = scale(data['el_power'], scaler = scaler_pow)
    scaler = [scaler_input, scaler_power]
    
    # append both scaled values with -1 (scaled as 0) by the lag
    input_scaled = np.append(input_scaled, np.full(lag, -1))
    power_scaled = np.append(power_scaled, np.full(lag, -1))
    
    
    # for each lag shift the input_voltage and the electric power accordingly. Create a new column in the data frame for each lag.
    for i in range(lag):
        df['input_voltage_delay_' + str(i)] = np.roll(input_scaled, i)[:len(data)]
        df['el_power_delay_' + str(i)] = np.roll(power_scaled, i)[:len(data)]
    
    diffs = df['el_power_delay_' + str(lag -1)] - df['el_power_delay_0']
    diffs[:lag] = 0
    y = df[['el_power_delay_0']] # in all cases use the current power as target 
    
    
    # parameter "all_lags" to get all or only two voltagess as model input
    if all_lags:
        X = df.filter(regex = ("input_voltage_delay_.*"))
    else:
        X = df[['input_voltage_delay_0', 'input_voltage_delay_' +str(lag-1)]]
    
    
    # parameter "hybrid" to include theoretical predictions in model input
    if hybrid:
        if not 'theor_predictions' in data:
            print('Have not found column "theor_predictions" in DataFrame. Please ensure to have theoretical predictions before running hybrid model.')
            return None
        theor_preds_scaled, scaler_theor = scale(data['theor_predictions'], scaler = scaler_pow)
        scaler.append(scaler_theor)
        X['theor_pred'] = theor_preds_scaled
       
    
    # possible variations of parameter differences
    if differences == 'add':
        X['difference'] = diffs
    elif differences == 'add_scaled':
        diffs_scaled, scaler_diffs = scale(diffs, scaler = scaler_pow)
        X['difference'] = diffs_scaled
        scaler.append(scaler_diffs)
    elif differences == 'predict':
        y['diff'] = diffs
    elif differences == 'predict_scaled':
        diffs_scaled, scaler_diffs = scale(diffs, scaler = scaler_pow)
        y['diff'] = diffs_scaled
        scaler.append(scaler_diffs)
    elif differences == 'input':
        X['input_difference'] = df['input_voltage_delay_' + str(lag -1)] - df['input_voltage_delay_0']
        X['input_difference'][:lag] = 0
    
    # reshape input values to follow LSTM tutorial
    X = X.values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = y.values
    y = y.reshape(y.shape[0], 1, y.shape[1])
    
    return scaler, X, y