import os
from sklearn import metrics
import pandas as pd
import numpy as np


#if folder does not exist, create such a folder
def check_folder(foldername):
    if os.path.exists(foldername):
        print('Folder already exists.')
    else:
        os.mkdir(foldername)
        print('Creation of directory %s successful.' % foldername)        

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

#open a CSV file within our structure
def open_CSV_file(filename, foldername, sep = "|", enc = "utf-8"):
    path = os.path.join(foldername, filename)
    df = pd.read_csv(path, delimiter = sep, encoding = enc)
    return df

# measure the difference of predictions and approximations. One can specify, which measures to use.
def measure_difference(
    true_values,
    approximations,
    R_SQUARED = True,
    RSME = True,
    AIC = False,
    MAE = True,
    MaxAE = False):
    
    values = true_values[~np.isnan(true_values)]
    approx = approximations[~np.isnan(approximations)]
    
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