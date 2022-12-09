# Author: Laura Kulowski

#'''

#Generate a synthetic dataset for our LSTM encoder-decoder
#We will consider a noisy sinusoidal curve 

#'''

import numpy as np
import torch
cuda = torch.device('cuda')

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
