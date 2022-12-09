# Author: Laura Kulowski
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import pandas as pd

def plot_test_results(lstm_model, Xtest, Ytest, Xtest_mean, Xtest_std, min_arr, max_arr,num_rows = 1):
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
  ow = Ytest.shape[0]
  
  # figure setup 
  num_cols = 1
  num_plots = num_rows * num_cols
  fig, ax = plt.subplots(num_rows, num_cols, figsize = (13, 15))

#  Y_test_pred_csv = np.zeros([25, nTestSample * 2])
#  Y_test_pred_csv = np.zeros([126, nTestSample * 4])
  Y_test_pred_csv = np.zeros([125, nTestSample * 4])

  for ii in range(nTestSample):
      vector = np.vectorize(np.float)
      ii = ii * 1
      #print(Xtest.shape)
      X_test_plt = Xtest[:, ii, :]
      Y_test_plt = Ytest[:, ii, :]

      Y_test_pred = lstm_model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor).cuda(), target_len = ow)
      print(Y_test_pred.shape)
      for i in range(4):
          # Y_test_pred[:,i] = (Y_test_pred[:,i] * Xtest_std[i]) + Xtest_mean[i]
          Y_test_pred[:, i] = ((Y_test_pred[:, i] * (max_arr[i] - min_arr[i]))+ min_arr[i])
      #print(dist_temp)

      #print(Y_test_plt)
      dist = np.sqrt((Y_test_pred[-1,0] - Y_test_plt[:, 0])**2 + (Y_test_pred[-1, 0] - Y_test_plt[:, 0])**2)      

      Y_test_pred_csv[:, ii * 2 : ii * 2 + 2] = Y_test_pred[:, 0:2]
#      Y_test_pred_csv[:, ii * 4: ii * 4 + 4] = Y_test_pred[:, 0:4]
      
      
 #     csv_name = 'csv/result' + str(ii) + '.csv'

#      np.savetxt(csv_name, Y_test_pred[:,0:2], delimiter=",")

      df = pd.DataFrame(Y_test_pred[:,0:2]) 
      #df.to_csv(csv_name, index=False)

      plt.plot(X_test_plt[:,0], X_test_plt[:,1])
      plt.plot(Y_test_pred[:,0], Y_test_pred[:,1])
      plt.plot(Y_test_plt[:,0], Y_test_plt[:,1])

#      plt.xlim([30, 90])
#      plt.ylim([30, 90])
      plt.xlim([45, 90])
      plt.ylim([35, 80])

      plt.tight_layout()
      #namefig = 'plots/' + str(ii) + '.png'
      #plt.savefig(namefig)

      plt.close() 
  print(Y_test_pred_csv)
  np.savetxt('csv/result.csv', Y_test_pred_csv, delimiter=",")
  
  return 


