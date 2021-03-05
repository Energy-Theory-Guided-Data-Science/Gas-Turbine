import numpy as np
import h5py
import lttb
import os

def downsample_data(time, values, sample_size = 12232):
    assert len(time) == len(values), "The data to be downsampled has to have the same length. Lengths in this case are " + len(time) + " for time and " + len(values) + "for values."
    data = np.array([time, values]).T
    while len(data.shape) != 2:
        data = data[0]
    downsampled_data = lttb.downsample(data, n_out = sample_size)
    assert downsampled_data.shape[0] == sample_size, "The downsampled value does not match the specified size. Somehow my code didn't work, I'm sorry."
    return downsampled_data[:,1]

#import all .matlab-files from data folder
def open_raw_mat_files(mat_raw_files, folder):
    raw_data = {}
    for file in mat_raw_files:
        path = os.path.join(folder, file)
        mat_file = h5py.File(path, 'r')
        group_data = open_file_with_headers(mat_file)
        raw_data[file[:-4]] = group_data
    return raw_data

# in case the matlab file has multiple headers, this method has to be executed
def open_file_with_headers(mat_file):
    group_data = {}
    for head in mat_file.keys():
        group_data[head] = np.squeeze(mat_file[head][:])
    return group_data

