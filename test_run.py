#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import sys
import pickle
import os
import generate_dataset
import lstm_encoder_decoder
import plotting_TestResult
import build_data
iw = 12
ow = 8
s = 1
num_feature = 4
file = '2_1_data_12_8_1_4'

# matplotlib.rcParams.update({'font.size': 17})
#----------------------------------------------------------------------------------------------------------------
# generate dataset for LSTM
# plot time series 
# plot time series with train/test split
#----------------------------------------------------------------------------------------------------------------
# window dataset
# set size of input/output windows 
#----------------------------------------------------------------------------------------------------------------


X_train, Y_train, X_test, Y_test = build_data.make_dataset(iw, ow , s, num_feature)

mean_arr, std_arr, max_arr, min_arr = build_data.parameters(X_test, num_feature)

# Normalization
for i in range(num_feature):
 #   Xtest[:,:,i] = (Xtest[:,:,i] - Xtrain_mean[i]) / Xtrain_std[i]
    X_test[:, :, i] = (X_test[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])


for i in range(num_feature):
  #  Ytest[:,:,i] = (Ytest[:,:,i] - Ytrain_mean[i]) / Ytrain_std[i]
    Y_test[:, :, i] = (Y_test[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])



#----------------------------------------------------------------------------------------------------------------

# Load LSTM model
path = "C:\\Users\\USER\\Desktop\\My_LSTM_example\\My_LSTM_example"
os.chdir(path)
filename = 'model/' + file +'.pkl'  #'model/inD_LSTM_ver02b_group02.pkl'  # inD_LSTM_ver02_group1_training
model = pickle.load(open(filename, 'rb'))

# plot predictions on train/test data
#plotting_TestResult.plot_test_results(model, Xtest, Ytest, Xtrain_mean, Xtrain_std)
plotting_TestResult.plot_test_results(model, X_test, Y_test, min_arr, max_arr, iw, ow, s, num_feature, file)

#-----------------------------------------------------------------------------------------------

