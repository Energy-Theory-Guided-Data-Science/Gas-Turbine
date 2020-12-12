import pandas as pd
import numpy as np
import os
import h5py
import lttb

def fillData(time_splits, values_splits):
    time = np.array(range(int(time_splits[-1])))
    values = np.array(range(int(time_splits[-1])))
    for i in range(len(time_splits)-1):
        lower = int(time_splits[i])
        upper = int(time_splits[i+1])
        v = values_splits[i]
        values[lower:upper] = v
    return time, values


def checkFolder(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)
        print('Creation of dircetory %s successful.' % foldername)
    else:
        print('Folder already exists.')

def openCSVFile(filename, foldername):
    path = os.path.join(foldername, filename)
    df = pd.read_csv(path, delimiter = "|", encoding = "utf-8")
    return df

def downsampleData(time, values, sample_size = 11290):
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