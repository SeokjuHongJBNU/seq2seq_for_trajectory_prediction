# Author: Laura Kulowski

#'''



#'''

import numpy as np
import torch
cuda = torch.device('cuda')

def train_test_split(y, split = 0.8):        # trian과 test를 나눠주는 함수

  '''
  
  split time series into train/test sets
  
  : param t:                      time array
  : para y:                       feature array
  : para split:                   percent of data to include in training set 
  : return t_train, y_train:      time/feature training and test sets;  
  :        t_test, y_test:        (shape: [# samples, 1])
  
  '''
  
  indx_split = int(split * len(y))              # 전체 데이터의 80%를 indx_split에 저장
  indx_train = np.arange(0, indx_split)         # 0부터 indx_split-1(80%) 까지 배열 생성
  indx_test = np.arange(indx_split, len(y))     # indx_split ~ 전체 데이터 수 까지의 배열 생성
  

  y_train = y[indx_train]                       # y_train = y값의 80%의 데이터

  y_train = y_train.reshape(-1, 2)              # y_train 값을 행 벡터 형태로 변환
  
 
  y_test = y[indx_test]                         # y_test = 20% 데이터
 
  y_test = y_test.reshape(-1, 2)                # 행벡터 형태로 변환
  
  return  y_train, y_test 
  


def windowed_dataset(y, input_window = 5, output_window = 1, stride = 1, num_features = 1):
  
    '''
    create a windowed dataset
    
    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model 
    : param output_window:    number of future y samples to predict  
    : param stide:            spacing between windows   
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''
  
    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1      # 샘플 수 = (전체 데이터 - input_window - output_window) // stride     + 1   
                                                    # // 은 결과를 int 형으로 계산

    X = np.zeros([input_window, num_samples, num_features])             # 배열생성(input window수, 샘플 수, 피쳐 수)
    Y = np.zeros([output_window, num_samples, num_features])            # 배열생성(output window수, 샘플 수, 피쳐 수)
    
    for ff in np.arange(num_features):                # ff = 0                  
        for ii in np.arange(num_samples):             # ii = 0 ~ 300 까지 301번
            
        
            start_x = stride * ii                     # 1. 5 * 0,   2. 5 * 1
            end_x = start_x + input_window            # 1. 0 + 80,  2. 5 + 80
            
            
            X[:, ii, ff] = y[start_x:end_x, ff]       # 1. X[:, 0, 0] = y_train[0:80, 0],   2. X[:, 1, 0] = y_train[5:85, 0]
            
          
            start_y = stride * ii + input_window      # 1. 5*0 + 80   2. 5*1 + 80
            end_y = start_y + output_window           # 1. 80 + 20    2. 85 + 20
            Y[:, ii, ff] = y[start_y:end_y, ff]       # 1. Y[:, 0, 0] = y_train[80:100, 0]   2. Y[:, 1, 0] = y_train[85:105, 0]
            
          

    return X, Y             # X (input_window, num_samples, num_features) 차원을 가짐
















def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
#    '''
#    convert numpy array to PyTorch tensor
    
#    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
#    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
#    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
#    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
#    : return X_train_torch, Y_train_torch,
#    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 

#    '''

    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

    X_train_torch = X_train_torch.cuda()
    Y_train_torch = Y_train_torch.cuda()
    X_test_torch = X_test_torch.cuda()
    Y_test_torch = Y_test_torch.cuda()

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
