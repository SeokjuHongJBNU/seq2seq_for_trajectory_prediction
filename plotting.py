# Author: Laura Kulowski
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_train_test_results(lstm_model, Xtrain, Ytrain, Xtest, Ytest, num_rows = 1):
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
  
  # input window size
  iw = Xtrain.shape[0]
  ow = Ytest.shape[0]

  # figure setup 
  num_cols = 1
  num_plots = num_rows * num_cols

  fig, ax = plt.subplots(num_rows, num_cols, figsize = (13, 15))

  # plot training/test predictions
  #for ii in range(num_rows):

      # train set
  for ii in range(30):
      ii1 = ii * 1000 #45100 30000
      X_train_plt = Xtrain[:, ii1, :]
      #print(X_train_plt_temp)
      #X_train_plt = X_train_plt_temp[::-1].copy()
      Y_train_pred = lstm_model.predict(torch.from_numpy(X_train_plt).type(torch.Tensor), target_len = ow)

      
      #tensor_mean = 35.3120
      #tensor_std = 7.1667
      #X_train_plt[:,0] = (X_train_plt[:,0] - tensor_mean) / tensor_std
      #X_train_plt[:,1] = (X_train_plt[:,1] - tensor_mean) / tensor_std
      #Y_train_pred[:,0] = (Y_train_pred[:,0] - tensor_mean) / tensor_std
      #Y_train_pred[:,1] = (Y_train_pred[:,1] - tensor_mean) / tensor_std

      x_mean = 69.1944
      x_std = 18.0391
      y_mean = 63.5480
      y_std = 17.3037
      X_train_plt[:,0] = (X_train_plt[:,0] * x_std) + x_mean
      X_train_plt[:,1] = (X_train_plt[:,1] * y_std) + y_mean
      Y_train_pred[:,0] = (Y_train_pred[:,0] * x_std) + x_mean
      Y_train_pred[:,1] = (Y_train_pred[:,1] * y_std) + y_mean

      plt.plot(X_train_plt[:,0], X_train_plt[:,1])
      plt.plot(Y_train_pred[:,0], Y_train_pred[:,1])
      plt.xlim([20, 90])
      plt.ylim([20, 90])
      #plt.xlim([-2, 2])
      #plt.ylim([-2, 2])
      print('last point')
      print(Y_train_pred[-1,0], Y_train_pred[-1,1])
      #namefig = 'plots/training_sample' + str(ii1) + '.png'
      #plt.savefig(namefig)
      plt.close()

      # test set
  for ii3 in range(33):
      ii2 = ii3 *2
      X_test_plt = Xtest[:, ii2, :]
      Y_test_pred = lstm_model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor), target_len = ow)

      #tensor_mean = 35.3120
      #tensor_std = 7.1667
      #X_test_plt[:,0] = (X_test_plt[:,0] * tensor_std) + tensor_mean
      #X_test_plt[:,1] = (X_test_plt[:,1] * tensor_std) + tensor_mean
      #Y_test_pred[:,0] = (Y_test_pred[:,0] * tensor_std) + tensor_mean
      #Y_test_pred[:,1] = (Y_test_pred[:,1] * tensor_std) + tensor_mean

      x_mean = 69.1944
      x_std = 18.0391
      y_mean = 63.5480
      y_std = 17.3037
      X_test_plt[:,0] = (X_test_plt[:,0] * x_std) + x_mean
      X_test_plt[:,1] = (X_test_plt[:,1] * y_std) + y_mean
      Y_test_pred[:,0] = (Y_test_pred[:,0] * x_std) + x_mean
      Y_test_pred[:,1] = (Y_test_pred[:,1] * y_std) + y_mean

      plt.plot(X_test_plt[:,0], X_test_plt[:,1])
      plt.plot(Y_test_pred[:,0], Y_test_pred[:,1])

      plt.xlim([20, 90])
      plt.ylim([20, 90])
      #plt.xlim([-2, 2])
      #plt.ylim([-2, 2])
      #plt.tight_layout()
      #namefig = 'plots/test_sample' + str(ii2) + '.png'
      #plt.savefig(namefig)
#       plt.savefig('plots/predictions2.png')
#      print('X_test_plt')
#      print(X_test_plt)
#      print('Y_test_pred')
#      print(Y_test_pred)
      plt.close() 
      
  return 


