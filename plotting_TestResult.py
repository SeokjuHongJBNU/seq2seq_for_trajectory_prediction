# Author: Laura Kulowski
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import pandas as pd
import os
import build_data

def plot_test_results(lstm_model, Xtest, Ytest, min_arr, max_arr, iw, ow, s, num_feature, file):
  '''
  plot examples of the lstm encoder-decoder evaluated on the training/test data
  
  : param lstm_model:     trained lstm encoder-decoder
  : param Xtrain:         np.array of windowed training input data
  : param Ytrain:         np.array of windowed training target data
  : param Xtest:          np.array of windowed test input data
  : param Ytest:          np.array of windowed test target data 
  : param num_rows:       number of training/test examples to plot
  : return:               num_rows x 2 plots; first column is training data predictions,
  :                       second column is test data predictions
  '''
  
  nTestSample = np.size(Xtest, 1)

  # input window size
  #iw = Xtrain.shape[0]
  ow = Ytest.shape[0]   # 40
  

  Y_test_pred_csv = np.zeros([ow, nTestSample * 2])     # [40, 1597*2]

  for ii in range(nTestSample): # 1597 만큼 반복
      #print(Xtest.shape)
      X_test_plt = Xtest[:, ii, :]
      Y_test_plt = Ytest[:, ii, :]
      
      Y_test_pred = lstm_model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor).cuda(), target_len = ow)
      #print(Y_test_pred.shape)
      for i in range(num_feature):
      # Y_test_pred[:,i] = (Y_test_pred[:,i] * Xtest_std[i]) + Xtest_mean[i]
          Y_test_pred[:, i] = ((Y_test_pred[:, i] * (max_arr[i] - min_arr[i]))+ min_arr[i])
      
      #print(Y_test_plt)
      
      Y_test_pred_csv[:, ii * 2 : ii * 2 + 2] = Y_test_pred[:, 0:2]

  
  print(Y_test_pred_csv)
  
  path_result = "C:\\Users\\USER\\Desktop\\My_LSTM_example\\My_LSTM_example\\csv"
  os.chdir(path_result)
  np.savetxt(file +'.csv', Y_test_pred_csv, delimiter=",")
  
  

  predict = pd.read_csv(file + '.csv', header = None, encoding='cp949').to_numpy()


  X_train, Y_train, X_test, Y_test = build_data.make_dataset(iw ,ow, s, num_feature)

  X_mean_arr, X_std_arr, X_max_arr, X_min_arr = build_data.parameters(X_test, num_feature)
  Y_mean_arr, Y_std_arr, Y_max_arr, Y_min_arr = build_data.parameters(Y_test, num_feature)
  P_mean_arr, P_std_arr, P_max_arr, P_min_arr = build_data.parameters(predict, 2)

  if num_feature == 4:
      
      X_max_arr = np.delete(X_max_arr, (2,3))
      X_min_arr = np.delete(X_min_arr, (2,3))
      Y_max_arr = np.delete(Y_max_arr, (2,3))
      Y_min_arr = np.delete(Y_min_arr, (2,3))

  elif num_feature == 3:
      X_max_arr = np.delete(X_max_arr, 2)
      X_min_arr = np.delete(X_min_arr, 2)
      Y_max_arr = np.delete(Y_max_arr, 2)
      Y_min_arr = np.delete(Y_min_arr, 2)
      
      
  for i in range(2):
      max_arr = np.maximum(X_max_arr, Y_max_arr, P_max_arr)
      min_arr = np.minimum(X_min_arr, Y_min_arr, P_min_arr)

  # path_plot = "C:\\Users\\USER\\Desktop\\My_LSTM_example\\My_LSTM_example\\plots"
  # os.chdir(path_plot)
  
  # os.makedirs(file,exist_ok=True)
  # os.chdir(''+file)

  for idx in range(Y_test.shape[1]):
      
       plt.plot(X_test[:,idx,0], X_test[:,idx,1], 'g', label='X_test(Input)')
       plt.plot(Y_test[:,idx,0], Y_test[:,idx,1], 'b', label='Y_test(Target)') 
       plt.plot(predict[:,idx*2], predict[:,idx*2+1], 'r', label='Predict')
     
       plt.legend(loc = 'lower left')
       plt.xlabel('X-Axis')
       plt.ylabel('Y-Axis')
       plt.xlim(min_arr[0], max_arr[0])
       plt.ylim(min_arr[1], max_arr[1])
       
       # namefig = 'Test_plot' + str(idx) + '.png'
       # plt.savefig(namefig)
      
       plt.pause(0.0001)
       plt.clf()

      

  
  return 


