import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment',  None) # 경고 off
import os
import generate_dataset


def make_dataset(iw ,ow ,s, num_feature):
    #make train dataset
    
    X_train = np.full((iw,1,num_feature),0)
    Y_train = np.full((ow,1,num_feature),0)
    path_train = "C:\\Users\\USER\\Desktop\\My_LSTM_example\\My_LSTM_example\\Dataset\\csv\\training_data"
    os.chdir(path_train)
    folders = os.listdir('C:/Users/USER/Desktop/My_LSTM_example/My_LSTM_example/Dataset/csv/training_data')
    select_feature = 0
    # print(folders)
    if num_feature == 3:
        select_feature = int(input('Vel(0) or Heading(1) : '))
    for files in folders: 
        
        #print(files)
        df = pd.read_csv(files, encoding='cp949')
       
        df_XY = df[['X (View Proj)', 'Y (View Proj)']]
        #print(files)  
        if num_feature == 2:
            df_XY = df_XY.dropna()
                
        elif num_feature == 3:
            if select_feature ==  0:
                df_XY = df[['X (View Proj)', 'Y (View Proj)', 'Length']]
                df_XY['Length'] = df_XY['Length'].str.replace('m', '')
                df_XY['Length'] = df_XY['Length'].str.replace('---', '')
                df_XY = df_XY.dropna(axis=0)
            elif select_feature == 1:
                df_XY = df[['X (View Proj)', 'Y (View Proj)', 'Heading']]
               
                df_XY = df_XY.dropna(axis=0)
                data = df_XY['Heading'].values
                yaw_rate = np.zeros((data.shape[0],1))
                for i in range(data.shape[0]):
                    if i == 0:
                        yaw_rate[i] = float(data[0])
                    else:
                        before_value = float(data[i-1])
                        current_value = float(data[i])
                        yaw_rate[i] = current_value-before_value
                    if yaw_rate[i] < -100:
                        yaw_rate[i] = yaw_rate[i] + 360
                    elif yaw_rate[i] > 100:
                        yaw_rate[i] = yaw_rate[i] - 360

                data = yaw_rate * 10
                data[0] = 0
                df_Heading = pd.DataFrame(data)

                df_XY = df_XY[['X (View Proj)', 'Y (View Proj)']]
                df_XY = pd.concat((df_XY, df_Heading), axis = 1)
                        
        else:
            df_XY = df[['X (View Proj)', 'Y (View Proj)', 'Length','Heading']]
            df_XY['Length'] = df_XY['Length'].str.replace('m', '')
            df_XY['Length'] = df_XY['Length'].str.replace('---', '')
            df_XY = df_XY.dropna(axis=0)
            data = df_XY['Heading'].values
            yaw_rate = np.zeros((data.shape[0],1))
            for i in range(data.shape[0]):
                if i == 0:
                    yaw_rate[i] = float(data[0])
                else:
                   before_value = float(data[i-1])
                   current_value = float(data[i])
                   yaw_rate[i] = current_value-before_value
                if yaw_rate[i] < -100:
                   yaw_rate[i] = yaw_rate[i] + 360
                elif yaw_rate[i] > 100:
                   yaw_rate[i] = yaw_rate[i] - 360

            data = yaw_rate * 10
            data[0] = 0
            df_Heading = pd.DataFrame(data)
            df_XY = df_XY[['X (View Proj)', 'Y (View Proj)', 'Length']]
            df_XY = pd.concat((df_XY, df_Heading), axis = 1)
            
        if df_XY['Y (View Proj)'].iloc[0] > df_XY['Y (View Proj)'].iloc[-1]:
            df_XY = df_XY.loc[::-1]
            
        trj_data = df_XY.to_numpy()
        snippet_X, snippet_Y = generate_dataset.windowed_dataset(trj_data, iw, ow, s, num_feature)
        
        X_train = np.concatenate((X_train, snippet_X), axis=1)
        Y_train = np.concatenate((Y_train, snippet_Y), axis=1)
    
    X_train = np.delete(X_train, 0, axis = 1)
    Y_train = np.delete(Y_train, 0, axis = 1)  
    #print(folders)
  ##------------------------------------------------------------------------------------------------------------------  
    # make test dataset
    
    X_test = np.full((iw,1,num_feature),0)
    Y_test = np.full((ow,1,num_feature),0)
    path_test = "C:\\Users\\USER\\Desktop\\My_LSTM_example\\My_LSTM_example\\Dataset\\csv\\test_data"
    os.chdir(path_test)
    folders2 = os.listdir('C:/Users/USER/Desktop/My_LSTM_example/My_LSTM_example/Dataset/csv/test_data')
    for files in folders2:
        df = pd.read_csv(files, encoding='cp949')
        
        if num_feature == 2:
            df_XY = df[['X (View Proj)', 'Y (View Proj)']]
            df_XY = df_XY.dropna()
            
        elif num_feature == 3:
            if select_feature ==  0:
                df_XY = df[['X (View Proj)', 'Y (View Proj)', 'Length']]
                df_XY['Length'] = df_XY['Length'].str.replace('m', '')
                df_XY['Length'] = df_XY['Length'].str.replace('---', '')
                df_XY = df_XY.dropna(axis=0)
            elif select_feature == 1:
                df_XY = df[['X (View Proj)', 'Y (View Proj)', 'Heading']]
               
                df_XY = df_XY.dropna(axis=0)
                data = df_XY['Heading'].values
                yaw_rate = np.zeros((data.shape[0],1))
                for i in range(data.shape[0]):
                    if i == 0:
                        yaw_rate[i] = float(data[0])
                    else:
                        before_value = float(data[i-1])
                        current_value = float(data[i])
                        yaw_rate[i] = current_value-before_value
                    if yaw_rate[i] < -100:
                        yaw_rate[i] = yaw_rate[i] + 360
                    elif yaw_rate[i] > 100:
                        yaw_rate[i] = yaw_rate[i] - 360

                data = yaw_rate * 10
                data[0] = 0
                df_Heading = pd.DataFrame(data)

                df_XY = df_XY[['X (View Proj)', 'Y (View Proj)']]
                df_XY = pd.concat((df_XY, df_Heading), axis = 1)
        else:
             df_XY = df[['X (View Proj)', 'Y (View Proj)', 'Length','Heading']]
             df_XY['Length'] = df_XY['Length'].str.replace('m', '')
             df_XY['Length'] = df_XY['Length'].str.replace('---', '')
             df_XY = df_XY.dropna(axis=0)
             data = df_XY['Heading'].values
             yaw_rate = np.zeros((data.shape[0],1))
             for i in range(data.shape[0]):
                 if i == 0:
                     yaw_rate[i] = float(data[0])
                 else:
                    before_value = float(data[i-1])
                    current_value = float(data[i])
                    yaw_rate[i] = current_value-before_value
                 if yaw_rate[i] < -100:
                    yaw_rate[i] = yaw_rate[i] + 360
                 elif yaw_rate[i] > 100:
                    yaw_rate[i] = yaw_rate[i] - 360

             data = yaw_rate * 10
             data[0] = 0
             df_Heading = pd.DataFrame(data)
 
             df_XY = df_XY[['X (View Proj)', 'Y (View Proj)', 'Length']]
             df_XY = pd.concat((df_XY, df_Heading), axis = 1)
        if df_XY['Y (View Proj)'].iloc[0] > df_XY['Y (View Proj)'].iloc[-1]:
            df_XY = df_XY.loc[::-1]
            
        trj_data = df_XY.to_numpy()
        snippet_X, snippet_Y = generate_dataset.windowed_dataset(trj_data, iw, ow, s, num_feature)
        
        X_test = np.concatenate((X_test, snippet_X), axis=1)
        Y_test = np.concatenate((Y_test, snippet_Y), axis=1)
    
    X_test = np.delete(X_test, 0, axis = 1)
    Y_test = np.delete(Y_test, 0, axis = 1)
    
    if (num_feature != 2 and select_feature != 1):
        X_train[:,:,2] = 10*X_train[:,:,2]
        Y_train[:,:,2] = 10*Y_train[:,:,2]
        X_test[:,:,2] = 10*X_test[:,:,2]
        Y_test[:,:,2] = 10*Y_test[:,:,2]
        
    
    return X_train, Y_train, X_test, Y_test



def parameters(X_train, num_feature):
   
    X_train_temp = X_train.reshape(-1,num_feature)
   # print(X_train_temp.shape)
    mean_arr = np.mean((X_train_temp), axis=0)
    std_arr = np.std((X_train_temp), axis=0)
    max_arr = np.max((X_train_temp),axis = 0)
    min_arr = np.min((X_train_temp),axis = 0)

    return mean_arr, std_arr, max_arr, min_arr

