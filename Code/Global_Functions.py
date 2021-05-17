################################################################################
######################## Import of libraries ###################################
################################################################################
import os
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py


################################################################################
####################### General functions ######################################
################################################################################

#if folder does not exist, create such a folder
def check_folder(foldername):
    if os.path.exists(foldername):
        print('Folder already exists.')
    else:
        os.makedirs(foldername)
        print('Creation of directory %s successful.' % foldername)        

# get color according to KIT color scheme
def get_color(color):
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

# use multiple experiments as training (or validation)
def use_multiple_experiments(experiments):
    df = pd.DataFrame()
    for i in experiments:
        df_ex = pd.DataFrame(i)
        df = pd.concat([df, df_ex], ignore_index=True)
    return df

# load synthetic data sets from folder
def load_synthetic(open_folder, length = None):
    experiments = []
    for root, dirs, files in os.walk(open_folder):
        for file in files:
            if file[-4:] == ".csv":
                experiments.append(open_CSV_file(file, open_folder))
                if len(experiments) == length:
                    return experiments
    return experiments

# create plot of two lines and save it to a specified image_folder. The lines stand for the true values and the predictions of the model respectively.
def create_prediction_plot(true_values, predictions, image_folder, title = "", specs = ""):
    fig = plt.figure(figsize = (15,10))
    plt.plot(true_values, color = get_color("grey"), label = "True")
    plt.plot(predictions, color = get_color("green"), label = "Predictions")
    plt.ylabel('Electric power [W]', fontsize = 18)
    plt.xlabel('Time [sec]', fontsize = 18)
    plt.legend()
    plt.title(title + specs, fontsize = 25)
    fig.tight_layout()
    plt.show()
    fig.savefig(image_folder + specs + "predictions.png")
    fig.savefig(image_folder + specs + "predictions.svg")

# measure the difference of predictions and approximations. One can specify, which measures to use.
def measure_difference(true_values, approximations,
                       R_SQUARED = True, RSME = True, AIC = False,
                       MAE = True, MaxAE = True, should_print = True):
    
    values = true_values[~np.isnan(true_values)]
    approx = approximations[~np.isnan(approximations)]
    
    min_length = min(len(values), len(approx))
    values = values[:min_length]
    approx = approx[:min_length]
    
    results = list()
    columns = list()
    
    if RSME:
        rms = metrics.mean_squared_error(values, approx, squared=False)
        results.append(rms)
        columns.append('RMSE')
        if should_print:
            print('The RMSE is %5.3f' %rms)
    if R_SQUARED:
        r2 = metrics.r2_score(values, approx)
        results.append(r2)
        columns.append('R2')
        if should_print:
            print('The R2-score is %5.3f' %r2)
    if MAE:
        mae = metrics.mean_absolute_error(values, approx)
        results.append(mae)
        columns.append('MAE')
        if should_print:
            print('The MAE is %5.3f' %mae)
    if MaxAE:
        maxae = metrics.max_error(values, approx)
        results.append(maxae)
        columns.append('MaxAE')
        if should_print:
            print('The MaxAE is %5.3f' %maxae)
    if AIC:
        resid = approx - values
        sse = sum(resid**2)
        aic = 2-2*np.log(sse)
        results.append(aic)
        columns.append('AIC')
        if should_print:
            print('The AIC is %5.3f' %aic)
    df = pd.DataFrame(columns = columns)
    df.loc[0] = results
    return df

################################################################################
########################## Mathematical functions ##############################
################################################################################

def exp_func(x, k, tau):
    return -k* np.exp((-1/tau)*x) +k

def quadr_func(x, p1, p2, p3):
    return p1 * x**2 + p2*x + p3

def linear_func(x, a, b):
    return a*x + b

################################################################################
################################ Open files ####################################
################################################################################

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

#open a CSV file within our structure
def open_CSV_file(filename, foldername, sep = "|", enc = "utf-8"):
    path = os.path.join(foldername, filename)
    df = pd.read_csv(path, delimiter = sep, encoding = enc)
    return df