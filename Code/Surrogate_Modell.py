import scipy
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import pandas as pd
import Global_Functions as gf


def linear_func(x, a, b):
    return a*x + b
	
def quadr_func(x, p1, p2, p3):
    return p1 * x**2 + p2*x + p3

def exp_func(x, k, tau):
    return -k* np.exp((-1/tau)*x) +k

def get_time_splits(df):
    time_splits = [x for x in range(len(df)-1) if df.iloc[x]['input_voltage'] != df.iloc[x+1]['input_voltage']]
    if len(time_splits) > 20:
        time_splits = time_splits[::2]
    time_splits.append(len(df))
    if time_splits[0] != 0:
        time_splits.insert(0,0)
    return time_splits

def approx(time, values):
    assert len(time) == len(values), 'time and values should have the same length'
    segment_start_value = values[0]
    
    param_bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
    para_fit, pcov = scipy.optimize.curve_fit(exp_func, time - time[0], values- segment_start_value, bounds=param_bounds)
    k_best = para_fit[0]
    tau_best = para_fit[1]
    return k_best, tau_best
	
	
def all_approximations(data_time, data_input, data_value, time_splits):
    fig,axs = plt.subplots(math.ceil(len(time_splits)/2), 2, figsize = (20,12))
    axs = axs.ravel()
    approximations = np.full(shape = len(data_time), fill_value= np.nan)
    parameters = []
    
    start = time.time()
    
    for t in range(len(time_splits) -1):
        cut_point_prior = math.ceil(time_splits[t])
        cut_point_post = math.ceil(time_splits[t+1])
        time_segment = np.array(data_time[cut_point_prior:cut_point_post])
        value_segment = np.array(data_value[cut_point_prior:cut_point_post])
        input_start = data_input[cut_point_prior]
        input_end = data_input[cut_point_post -1]
       
        k_best, tau_best = approx(time_segment,
                                  value_segment)
        segment_approximation = exp_func(time_segment-time_segment[0], k_best, tau_best) + value_segment[0]
        approximations[cut_point_prior:cut_point_post] = segment_approximation
        prs = [input_start, input_end, k_best, tau_best]
        parameters = np.concatenate((parameters, prs), axis = 0)
        
        axs[t].plot(time_segment, value_segment)
        axs[t].plot(time_segment, segment_approximation)
        
        
    end = time.time()
    duration = end - start
    axs[-1].plot(data_time, data_value)
    axs[-1].plot(data_time, approximations)
    
    return approximations, parameters.reshape(len(time_splits) -1, 4), duration
	
	
def create_dataframe_for_parameters(params):
    df_params = pd.DataFrame(params)
    df_params.columns = ['input_start', 'input_end', 'k_best', 'tau_best']
    df_params['input_diff'] = [df_params.iloc[x]['input_end'] - df_params.iloc[x]['input_start'] for x in range(len(df_params))]
    return df_params
	
def parameter_modelling(df_params, func_k, func_tau, name, image_folder):
    prm_fit_k, pcov_k = scipy.optimize.curve_fit(func_k, df_params['input_diff'], df_params['k_best'])
    prm_fit_tau, pcov_tau = scipy.optimize.curve_fit(func_tau, df_params['input_diff'], df_params['tau_best'])
    
    tau1, tau2, tau3 = prm_fit_tau
    k1, k2 = prm_fit_k
    
    input_default = np.arange(-7.0, 7-0, 0.2)
    
    fig,axs = plt.subplots(1, 2, figsize = (8,5))
    axs = axs.ravel()
    axs[1].scatter(df_params['input_diff'], df_params['tau_best'], label = 'tau')
    axs[1].plot(input_default, func_tau(input_default, tau1, tau2, tau3), color = gf.get_color("orange"))
    axs[1].set_title('tau over input difference for ' + name)
    axs[1].set_ylabel('tau')
    axs[1].set_xlabel('input difference')
    axs[0].scatter(df_params['input_diff'], df_params['k_best'], label ='k')
    axs[0].plot(input_default, func_k(input_default, k1, k2), color = gf.get_color("orange"))
    axs[0].set_title('k over input difference for ' + name)
    axs[0].set_ylabel('k')
    axs[0].set_xlabel('input difference')
    
    plt.savefig(image_folder + name + "_k_tau.png")
    plt.savefig(image_folder + name + "_k_tau.svg")
    
    return prm_fit_k, prm_fit_tau
	
def predict_other_experiment(parameters_k, parameters_tau, data, header):
    start = time.time()
    time_splits = get_time_splits(data)
    data_time = data['time']
    data_value = data[header]
    data_input = data['input_voltage']
    prm_k = np.empty(3)
    prm_tau = np.empty(3)
    prm_k[0], prm_k[1] = parameters_k
    prm_tau[0], prm_tau[1], prm_tau[2] = parameters_tau

    approximations = np.full(shape = len(data_time), fill_value= np.nan)
    
    for t in range(len(time_splits) -1):
        cut_point_prior = math.ceil(time_splits[t])
        cut_point_post = math.ceil(time_splits[t+1])
        time_segment = np.array(data_time[cut_point_prior:cut_point_post])
        value_segment = np.array(data_value[cut_point_prior:cut_point_post])
        input_start = data_input[cut_point_prior]
        input_end = data_input[cut_point_post -1]
        input_diff = input_end - input_start
        
        k = linear_func(input_diff, prm_k[0], prm_k[1])
        tau = quadr_func(input_diff, prm_tau[0], prm_tau[1], prm_tau[2])
        
        segment_approximation = exp_func(time_segment-time_segment[0], k, tau) + value_segment[0]
        approximations[cut_point_prior:cut_point_post] = segment_approximation

    end = time.time()
    duration = end-start
    return approximations, duration