#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import sys
import pickle

import generate_dataset
import lstm_encoder_decoder
import plotting_TestResult

import scipy.io


# matplotlib.rcParams.update({'font.size': 17})
#----------------------------------------------------------------------------------------------------------------
# generate dataset for LSTM
# plot time series 
# plot time series with train/test split
#----------------------------------------------------------------------------------------------------------------
# window dataset
# set size of input/output windows 
#----------------------------------------------------------------------------------------------------------------

trj_data_mat = scipy.io.loadmat('data/inD_LSTM_ver02_group1_validation_2sec.mat')  # ValidationData_TimeHist_1to1_211026

trj_data = list(trj_data_mat.items())
trj_data_arr = np.array(trj_data)
print(trj_data_arr[9,0])
print(trj_data_arr[3,1].shape)
Xtest = trj_data_arr[3,1]
Ytest = trj_data_arr[4,1]
mean = trj_data_arr[7,1]
std = trj_data_arr[8,1]
min_arr_temp = trj_data_arr[9,1]
max_arr_temp = trj_data_arr[10,1]

Xtrain_mean = mean[0]
Xtrain_std = std[0]
Ytrain_mean = Xtrain_mean
Ytrain_std = Xtrain_std
ow = Ytest.shape[0]

min_arr = min_arr_temp[0]
max_arr = max_arr_temp[0]
print(min_arr)
# Normalization
for i in range(4):
 #   Xtest[:,:,i] = (Xtest[:,:,i] - Xtrain_mean[i]) / Xtrain_std[i]
    Xtest[:, :, i] = (Xtest[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])

for i in range(4):
  #  Ytest[:,:,i] = (Ytest[:,:,i] - Ytrain_mean[i]) / Ytrain_std[i]
    Ytest[:, :, i] = (Ytest[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])

#----------------------------------------------------------------------------------------------------------------

# Load LSTM model
filename = 'model/inD_LSTM_ver02b_group02.pkl'  # inD_LSTM_ver02_group1_training
model = pickle.load(open(filename, 'rb'))

# plot predictions on train/test data
#plotting_TestResult.plot_test_results(model, Xtest, Ytest, Xtrain_mean, Xtrain_std)
plotting_TestResult.plot_test_results(model, Xtest, Ytest, Xtrain_mean, Xtrain_std, min_arr, max_arr)

plt.close('all')

