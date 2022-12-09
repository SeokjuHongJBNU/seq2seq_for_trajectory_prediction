
%% ���ϸ�: inD_LSTM_ver02_group1_validation_padding_5hz2
% 5Hz �������� 0 ���� padding
sample_rate = 25; % [25 hz ������]

for i = 1:76
    zero_pad_frq = 6;
    b = round(sample_rate / zero_pad_frq);   % 5Hz: 5, 1Hz: 25
    r = rem(i,b);
    
    if r == 0
        Test_X_stack(i,:,:) = 0;
%         Test_Y_stack(i,:,:) = 0;
    end
    
end

% %% ���ϸ�: inD_LSTM_ver02_group1_validation_padding_5hz
% % 5Hz �������� �����͸� ���ܵΰ� �������� 0�� padding (�������� ����� �����߸���)
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
