import numpy as np
import random
import os, errno
import sys
from tqdm import trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

cuda = torch.device('cuda')

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 3):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)

    def forward(self, x_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 3):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        #print(input_size)
        
        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)
        #print(self.encoder)
        #print(self.decoder)

    def train_model(self, input_tensor_temp, target_tensor_temp, X_test_input, Y_test_GT, mean_arr, std_arr, n_epochs, target_len, batch_size, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.0001, dynamic_tf = False):

        ''' (X_train, Y_train, X_test, Y_test, mean_arr, std_arr, n_epochs = 3000, target_len = ow, batch_size = 500, training_prediction = 'teacher_forcing', teacher_forcing_ratio = 0.5, learning_rate = 0.001, dynamic_tf = True)
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs 
        : param target_len:                number of values to predict 
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''

        input_tensor = input_tensor_temp.cuda()
        target_tensor = target_tensor_temp.cuda()

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        val_losses = np.full(n_epochs, np.nan)
        losses2 = np.full(n_epochs, np.nan)
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss().cuda()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)
        iter_break = 0

        with trange(n_epochs) as tr:
            for it in tr:
                
                batch_loss = 0.
                batch_loss2 = 0.
                batch_loss_tf = 0
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0
                b1 = 0
                for b in range(n_batches):
                    # select data
                    b1 = b * batch_size
                   # print([b1, b1+batch_size])
                    #print(b1 + batch_size)
                    input_batch = input_tensor[:, b1: b1 + batch_size, :]
                    target_batch = target_tensor[:, b1: b1 + batch_size, :]
                    #b1 = b + batch_size
                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                    
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # outputs[:,:,0:2]
                    loss = criterion(outputs[:,:,0:4].to(device='cuda'), target_batch[:,:,0:4].to(device='cuda')).to(device='cuda')
                    loss2 = criterion(outputs[:,:,0:2].to(device='cuda'), target_batch[:,:,0:2].to(device='cuda')).to(device='cuda')
                    batch_loss += loss.item()
                    batch_loss2 += loss2.item()
                    # print(loss)
                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches
                batch_loss2 /= n_batches
                losses[it] =  batch_loss
                losses2[it] = batch_loss2
                #plt.savefig('batch_plot')
                #plt.close() 

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02 
                loss_temp2 = 0
                intv = 50
                for i in range(0, X_test_input.shape[1], intv):  # 25 # val loss 구하는 과정
                    Y_test_pred = self.predict(X_test_input[:, i ,:], target_len)
                    #print(Y_test_pred.shape)
                    #loss_temp = criterion(Y_test_GT[:, i ,0:2], torch.from_numpy(Y_test_pred[:, 0:2]).type(torch.Tensor).cuda()).cpu().detach().numpy()
                    loss_temp = criterion(Y_test_GT[:, i, 0:4].cuda(), torch.from_numpy(Y_test_pred[:, 0:4]).type(
                        torch.Tensor).cuda())
                    loss_temp2 += loss_temp.item()
                #l_mean = (loss_temp2 / X_test_input.shape[1])
                l_mean = (loss_temp2 / (X_test_input.shape[1]/intv))
                val_losses[it] = l_mean
                # progress bar 
                print('')
                tr.set_postfix(tr_loss="{0:.5f}".format(batch_loss), val_loss="{0:.5f}".format(l_mean))

                # if l_mean < 0.00595:
                #     iter_break = iter_break + 1
                #     print(iter_break)

                # if iter_break >= 5:
                #     break
                
        
        return losses, val_losses, losses2

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        # encode input_tensor
        #print(input_tensor)
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        #print('input_tensor')
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])
        
        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
        np_outputs = outputs.detach().numpy()

        
        return np_outputs


