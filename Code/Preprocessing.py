import pandas as pd
import numpy as np
import os
import h5py
import lttb

OPEN_FOLDER = "../Data/Raw_Data/" # where are the raw matlab files?
SAVE_FOLDER = "../Data/Temp_Data/" # where do you want to save the .csv files


def createFolder(foldername):
	if not os.path.exists(foldername):
		os.mkdir(foldername)
		print('Creation of dircetory %s successful.' % foldername)

def openCSVFile(filename, foldername):
    path = os.path.join(foldername, filename)
    df = pd.read_csv(path, delimiter = "|", encoding = "utf-8")
    return df
	
	
# in case the matlab file has multiple headers, this method has to be executed
def openFileWithMultipleHeaders(path, mat_file):
    group_data = {}
    for head in mat_file.keys():
        group_data[head] = np.squeeze(mat_file[head][:])
    return group_data
	
	
	
#import all .matlab-files from data folder
def openMatfiles():
    raw_data = {}
    mat_raw_files = (file for file in os.listdir(OPEN_FOLDER) if file[-4:] == '.mat' and "_Test_" in file)
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
	
	
# downsample the data for a given sheet based on the Least Traingle Three Buckets algorithm
# The sample_size has to be given globally
def downsampleData(time, values, sample_size):
    assert len(time) == len(values)
    data = np.array([time, values]).T
    downsampled_data = lttb.downsample(data, n_out = sample_size)
    assert downsampled_data.shape[0] == sample_size
    return downsampled_data


# downsample all columns of a given data sheet
# it is neccessary to know the head of the time column
def downsampleSheet(sheet, time_head, sample_size):
    data_down = {}
    time_downsampled = np.arange(sample_size)
    sheet_time = sheet[time_head]
    if 't_2A_el' in time_head:
        sheet_time = sheet[time_head][::1000]
        for head in sheet.keys():
            sheet[head] = sheet[head][::1000]
    for head in sheet.keys():
        data_down[head] = downsampleData(sheet_time, sheet[head], sample_size)[:,1]
        if head == time_head:
            data_down[head] = time_downsampled
    return data_down
	
	
# prepare and downsample all sheets given by our expert
def prepareSheets(raw_data, sample_size):
    data = {}
    switcher_time = {
        'Daten_Test_ID_4b_1B_el': 't_1B_el',
        'Daten_Test_ID_4b_1B_th': 't_1B_th',
        'Daten_Test_ID_4b_2A_el_1': 't_2A_el_1',
        'Daten_Test_ID_4b_2A_el_2': 't_2A_el_2',
        'Daten_Test_ID_4b_2A_th': 't_2A_th',
        'Drehzahldaten_Test_ID_4b': 't_nsoll_stil',
        'Leistungdaten_Test_ID_4b': 't_elstil'
    }
    exclude = ['Drehzahldaten_Test_ID_4b', 'Leistungdaten_Test_ID_4b']
    for head in [x for x in raw_data.keys() if x not in exclude]:
        print(head)
        data[head] = downsampleSheet(raw_data[head], switcher_time.get(head), sample_size)
    for head in exclude:
        data[head] = raw_data[head]
    return data