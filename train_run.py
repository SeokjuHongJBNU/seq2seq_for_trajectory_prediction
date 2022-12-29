#!/usr/bin/env python3


import pickle
import os
import torch
import numpy as np
import pandas as pd
import generate_dataset
import lstm_encoder_decoder
import build_data
import matplotlib.pyplot as plt

print(torch.cuda.get_device_name(0))

device = torch.cuda.device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
cuda = torch.device('cuda')
#--------------------------------------------------------------------------------------------------------------------------

# 1. Data Import
iw = 12
ow = 8
s = 1
num_feature = 4
file = '2_1_data_12_8_1_4'


X_train, Y_train, X_test, Y_test = build_data.make_dataset(iw, ow, s, num_feature)


mean_arr, std_arr, max_arr, min_arr = build_data.parameters(X_train, num_feature)



# 2. Parameter
# ow = 10  # Numer of output sequence setting (50 sequence in LSTM output)



# 3. Variables - Data


# 4. Normalization
for i in range(num_feature):
    X_train[:,:,i] = (X_train[:,:,i] - min_arr[i]) / (max_arr[i] - min_arr[i])
    Y_train[:, :, i] = (Y_train[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])
    


mean_arr, std_arr, max_arr, min_arr = build_data.parameters(X_test, num_feature)

for i in range(num_feature):
    X_test[:, :, i] = (X_test[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])
    Y_test[:, :, i] = (Y_test[:, :, i] - min_arr[i]) / (max_arr[i] - min_arr[i])

#    Xtrain[:,:,i] = (Xtrain[:,:,i] - mean_arr[i]) / std_arr[i]
#    Ytrain[:,:,i] = (Ytrain[:,:,i] - mean_arr[i]) / std_arr[i]
#    Xtest[:,:,i] = (Xtest[:,:,i] - mean_arr[i]) / std_arr[i]
#    Ytest[:,:,i] = (Ytest[:,:,i] - mean_arr[i]) / std_arr[i]

#----------------------------------------------------------------------------------------------------------------
# 5. convert windowed data from np.array to PyTorch tensor
Xtrain, Ytrain, Xtest, Ytest = generate_dataset.numpy_to_torch(X_train, Y_train, X_test, Y_test)

# 6. specify model parameters and train
model = lstm_encoder_decoder.lstm_seq2seq(input_size = X_train.shape[2], hidden_size = 100)
model = model.cuda()
#model = model
loss, val_losses, losses2  = model.train_model(Xtrain, Ytrain, Xtest, Ytest, mean_arr, std_arr, n_epochs = 100, target_len = ow, batch_size = 100, training_prediction = 'teacher_forcing', teacher_forcing_ratio = 0.5, learning_rate = 0.001, dynamic_tf = True)

plt.plot(np.arange(val_losses.shape[0]),val_losses)
plt.plot(np.arange(losses2.shape[0]),losses2)

losses2 = losses2.reshape(-1,1)
val_losses = val_losses.reshape(-1,1)
total_losses = np.concatenate((losses2, val_losses), axis = 1)

path_result = "C:\\Users\\USER\\Desktop\\My_LSTM_example\\My_LSTM_example\\Total_losses"
os.chdir(path_result)
np.savetxt(file +'.csv', total_losses, delimiter=",")

  
# Load model
path = "C:\\Users\\USER\\Desktop\\My_LSTM_example\\My_LSTM_example"
os.chdir(path)
filename = 'model/'+ file +'.pkl'
pickle.dump(model, open(filename, 'wb'))

