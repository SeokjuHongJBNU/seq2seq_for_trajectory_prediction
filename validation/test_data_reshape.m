
%% 파일명: inD_LSTM_ver02_group1_validation_padding_5hz2
% 5Hz 간격으로 0 값을 padding
sample_rate = 25; % [25 hz 고정값]

for i = 1:76
    zero_pad_frq = 6;
    b = round(sample_rate / zero_pad_frq);   % 5Hz: 5, 1Hz: 25
    r = rem(i,b);
    
    if r == 0
        Test_X_stack(i,:,:) = 0;
%         Test_Y_stack(i,:,:) = 0;
    end
    
end

% %% 파일명: inD_LSTM_ver02_group1_validation_padding_5hz
% % 5Hz 간격으로 데이터를 남겨두고 나머지는 0로 padding (오리지널 결과값 망가뜨리기)
% for i = 1:76
%     b = 5;
%     r = rem(i,b);
%     
%     if r ~= 0
%         Test_X_stack(i,:,:) = 0;
%         Test_Y_stack(i,:,:) = 0;
%     end
%     
% end
% 
% % Test_X_stack(:,:,3)
