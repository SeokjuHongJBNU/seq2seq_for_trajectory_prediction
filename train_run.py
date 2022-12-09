#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle

import generate_dataset
import lstm_encoder_decoder

import scipy.io
import torch

print(torch.cuda.get_device_name(0))

device = torch.cuda.device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
cuda = torch.device('cuda')
#--------------------------------------------------------------------------------------------------------------------------

# 1. Data Import
trj_data_mat = scipy.io.loadmat('data/inD_LSTM_ver02_group1_training.mat') # TrainingData_new_time_total, LSTM_DataSet_ver03
trj_data = list(trj_data_mat.items())
trj_data_arr = np.array(trj_data)

# 2. Parameter
# ow = 10  # Numer of output sequence setting (50 sequence in LSTM output)

# trj_data_arr[0] : header  // trj_data_arr[1] : version info and data type   // trj_data_arr[2] :  // trj_data_arr[3] : array data(including var_names)
# Test_X: trj_data_arr[3,1],  Test_Y: trj_data_arr[4,1], Train_X: trj_data_arr[5,1],  trj_data_arr[6,1]
#var_name = trj_data_arr[0]
#trj_data_values = trj_data_arr[3]

# 3. Variables - Data
print('data new')
Xtrain = trj_data_arr[5,1]
Ytrain = trj_data_arr[6,1]
Xtest = trj_data_arr[3,1]
Ytest = trj_data_arr[4,1]
c_mean = trj_data_arr[7,1]
c_std = trj_data_arr[8,1]

print(trj_data_arr[9,0])
c_min = trj_data_arr[9,1]
c_max = trj_data_arr[10,1]

ow = Ytrain.shape[0]

mean_arr = c_mean[0]
std_arr = c_std[0]

min_arr = c_min[0]
max_arr = c_max[0]

# 4. Normalization
for i in range(4):
    Xtrain[:,:,i] = (Xtrain[:,:,i] - min_arr[i]) / (max_arr[i] - min_arr[i])
    Ytrain[:, :, i] = (Ytrain[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])
    Xtest[:, :, i] = (Xtest[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])
    Ytest[:, :, i] = (Ytest[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])

#    Xtrain[:,:,i] = (Xtrain[:,:,i] - mean_arr[i]) / std_arr[i]
#    Ytrain[:,:,i] = (Ytrain[:,:,i] - mean_arr[i]) / std_arr[i]
#    Xtest[:,:,i] = (Xtest[:,:,i] - mean_arr[i]) / std_arr[i]
#    Ytest[:,:,i] = (Ytest[:,:,i] - mean_arr[i]) / std_arr[i]

#----------------------------------------------------------------------------------------------------------------
# 5. convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

# 6. specify model parameters and train
model = lstm_encoder_decoder.lstm_seq2seq(input_size = X_train.shape[2], hidden_size = 100)
model = model.cuda()
#model = model
loss = model.train_model(X_train, Y_train, X_test, Y_test, mean_arr, std_arr, n_epochs = 2000, target_len = ow, batch_size = 100, training_prediction = 'teacher_forcing', teacher_forcing_ratio = 0.5, learning_rate = 0.0001, dynamic_tf = True)

# Load model
filename = 'model/inD_LSTM_ver02_group1_training_4feat.pkl'
pickle.dump(model, open(filename, 'wb'))

