import pandas as pd
import numpy as np
import os
import h5py
import lttb
from sklearn import metrics

def getColor(color):
    switcher = {
        'green' : '#009682',
        'blue' : '#4664AA',
        'orange' : '#DF9B1B',
        'lightgreen': '#77A200',
        'yellow': '#FCE500',
        'red': '#A22223',
        'purple': '#A3107C',
        'brown': '#A7822E',
        'cyan': '#079EDE',
        'black': '#000000',
        'grey': '#666666',
    }
    return switcher.get(color, 'The currently allowed colors are green, blue, orange, lightgreen, yellow, red, purple, brown and cyan.')

def fillData(time_splits, values_splits):
    time = np.array(range(int(time_splits[-1])))
    values = np.array(range(int(time_splits[-1])))
    for i in range(len(time_splits)-1):
        lower = int(time_splits[i])
        upper = int(time_splits[i+1])
        v = values_splits[i]
        values[lower:upper] = v
    return time, values

def applyConstraint(time_splits, values_splits, m = 2.16):
    time = np.array(range(int(time_splits[-1])))
    values = np.array(range(int(time_splits[-1])))
    for i in range(len(time_splits)-1):
        lower = int(time_splits[i])
        upper = int(time_splits[i+1])
        v_low = values_splits[i]
        v_up = values_splits[i+1]
        slope = min(m, ((v_up - v_low)/(upper-lower)))
        for t in range(upper-lower):
            values[lower + t] = v_low + slope *t
    return time, values


def checkFolder(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)
        print('Creation of directory %s successful.' % foldername)
    else:
        print('Folder already exists.')

def openCSVFile(filename, foldername):
    path = os.path.join(foldername, filename)
    df = pd.read_csv(path, delimiter = "|", encoding = "utf-8")
    return df

def downsampleData(time, values, sample_size = 12232):
    assert len(time) == len(values)
    data = np.array([time, values]).T
    while len(data.shape) != 2:
        data = data[0]
    downsampled_data = lttb.downsample(data, n_out = sample_size)
    assert downsampled_data.shape[0] == sample_size
    return downsampled_data

#import all .matlab-files from data folder
def openMatfiles(OPEN_FOLDER):
    raw_data = {}
    mat_raw_files = (file for file in os.listdir(OPEN_FOLDER) if file[-4:] == '.mat' and "daten_Test_" in file)
    for file in mat_raw_files:
        print(file)
        path = os.path.join(OPEN_FOLDER, file)
        mat_file = h5py.File(path, 'r')
        if len(mat_file.keys()) >1:
            group_data = openFileWithMultipleHeaders(path, mat_file)
        else:
            group_data = mat_file[:]
        raw_data[file[:-4]] = group_data
    return raw_data

# in case the matlab file has multiple headers, this method has to be executed
def openFileWithMultipleHeaders(path, mat_file):
    group_data = {}
    for head in mat_file.keys():
        group_data[head] = np.squeeze(mat_file[head][:])
    return group_data

def measureDifference(data, value_header, approx_header):
    R_SQUARED = True
    RSME = True
    AIC = False
    MAE = True
    MaxAE = True
    
    data = data[data[approx_header].notnull()]
    values = data[value_header]
    approx = data[approx_header]
    
    if RSME:
        rms = metrics.mean_squared_error(values, approx, squared=False)
        print('The RMSE is %5.3f' %rms)
    if R_SQUARED:
        r2 = metrics.r2_score(values, approx)
        print('The R2-score is %5.3f' %r2)
    if MAE:
        mae = metrics.mean_absolute_error(values, approx)
        print('The MAE is %5.3f' %mae)
    if MaxAE:
        maxae = metrics.max_error(values, approx)
        print('The MaxAE is %5.3f' %maxae)
    if AIC:
        resid = approx - values
        sse = sum(resid**2)
        aic = 2-2*np.log(sse)
        print('The AIC is %5.3f' %aic)