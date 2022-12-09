
data_x = [Train_X_stack, Test_X_stack];
data_y = [Train_Y_stack, Test_Y_stack];

%% Normalization params. %%
Train = cat(1, data_x, data_y);
nFeatures = size(Train,3);
c_mean = []; c_std = [];
c_min = []; c_max = [];

for i = 1 : nFeatures

    x_data = Train(:,:,i);
    c0_mean = mean(x_data(:), 1);
    c0_std = std(x_data(:),1);  

    c_mean = [c_mean, c0_mean];
    c_std = [c_std, c0_std];
    
    c_min_temp = min(x_data(:));
    c_max_temp = max(x_data(:));
    
    c_min = [c_min , c_min_temp];
    c_max = [c_max , c_max_temp];
    
end