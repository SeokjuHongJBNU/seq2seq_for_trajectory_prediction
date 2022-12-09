close all;  %clear; clc;

frame_sec = 1/25; % 25 frame per sec
fr_2sec = 2 / frame_sec;
fr_3sec = 3 / frame_sec;
fr_4sec = 4 / frame_sec;
fr_5sec = 5 / frame_sec;

hist_length = 3 / frame_sec;

%% 예측결과 csv 로딩
pred_csv = "result.csv";
result_csv_root = strcat("C:\TrajectoryPrediction\inD_LSTM\csv\", pred_csv);
pred = load(result_csv_root);

%% 원본데이터 로딩
ValidationData = "inD_LSTM_ver02_group1_validation_2sec";  % inD_LSTM_ver02_validation  inD_LSTM_ver01
ValidationSample_root = strcat("C:\TrajectoryPrediction\inD_LSTM\data\", ValidationData, ".mat");
load(ValidationSample_root, 'Test_X_stack', 'Test_Y_stack');

n4 = size(pred,2) * 0.5;
error_Lat = zeros(n4, 4);
error_P2P = zeros(n4, 4);

for i2 = 1 : n4
  
    pred_xy_2sec = pred(fr_2sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_3sec = pred(fr_3sec, (2*i2-1):(2*i2));
    pred_xy_4sec = pred(fr_4sec, (2*i2-1):(2*i2));
    pred_xy_5sec = pred(fr_5sec, (2*i2-1):(2*i2));

    pred_pos = [pred_xy_2sec; pred_xy_3sec; pred_xy_4sec; pred_xy_5sec];
    
    x_trans_hist = Test_X_stack(:,i2, 1);
    y_trans_hist = Test_X_stack(:,i2, 2);
    x_trans_future_temp = Test_Y_stack(:,i2, 1);
    y_trans_future_temp = Test_Y_stack(:,i2, 2);
    x_trans_future = nonzeros(x_trans_future_temp);
    y_trans_future = nonzeros(y_trans_future_temp);
    
    x_trans = [x_trans_hist; x_trans_future];
    y_trans = [y_trans_hist; y_trans_future];
    X_in_temp = [x_trans, y_trans];
    X_in = X_in_temp(1:10:end,:);

    figure;
    plot(x_trans_hist, y_trans_hist, 'b', 'linewidth', 2);  hold on; grid on; axis([50 200 -120 0]);
    plot(x_trans_future, y_trans_future, 'r');
    plot(pred(:,(2*i2-1)), pred(:,(2*i2)), 'k', 'linewidth', 1.1); % hold on; grid on; axis([30 90 30 90]);
    
    [xy, e_ho_2sec, t_a] = distance2curve([X_in(:,1), X_in(:,2)], [pred_pos(1,1) pred_pos(1,2)], 'spline');
    [xy, e_ho_3sec, t_a] = distance2curve([X_in(:,1), X_in(:,2)], [pred_pos(2,1) pred_pos(2,2)], 'spline');
    [xy, e_ho_4sec, t_a] = distance2curve([X_in(:,1), X_in(:,2)], [pred_pos(3,1) pred_pos(3,2)], 'spline');
    [xy, e_ho_5sec, t_a] = distance2curve([X_in(:,1), X_in(:,2)], [pred_pos(4,1) pred_pos(4,2)], 'spline');
    
    error_Lat(i2, 1) = e_ho_2sec;
    error_Lat(i2, 2) = e_ho_3sec;
    error_Lat(i2, 3) = e_ho_4sec;
    error_Lat(i2, 4) = e_ho_5sec;

    %% Point-to-point 평가
    Pos_Current_GT_XY = [x_trans_hist(end), y_trans_hist(end)];
    plot(Pos_Current_GT_XY(1), Pos_Current_GT_XY(2), 'o', 'color', 'k');
    idx_current = find(y_trans >= Pos_Current_GT_XY(2), 1);
    
    idx_t_target_2s = hist_length + fr_2sec;
    idx_t_target_3s = hist_length + fr_3sec;
    idx_t_target_4s = hist_length + fr_4sec; 
    idx_t_target_5s = hist_length + fr_5sec;
    
    xy_target_2s = [x_trans(idx_t_target_2s), y_trans(idx_t_target_2s)];
    xy_target_3s = [x_trans(idx_t_target_3s), y_trans(idx_t_target_3s)];
    xy_target_4s = [x_trans(idx_t_target_4s), y_trans(idx_t_target_4s)];
    xy_target_5s = [x_trans(idx_t_target_5s), y_trans(idx_t_target_5s)];
    plot(x_trans(idx_t_target_5s), y_trans(idx_t_target_5s), 'o');
    error_p2p_2s = NaN;
    error_p2p_3s = NaN; 
    error_p2p_4s = NaN;
    error_p2p_5s = NaN;
    
    if ~isempty(pred_xy_2sec) && ~isempty(xy_target_2s)
        error_p2p_2s_temp = distance(pred_xy_2sec, xy_target_2s);
        error_p2p_2s = sqrt((pred_xy_2sec(1) - xy_target_2s(1))^2 + (pred_xy_2sec(2) - xy_target_2s(2))^2);
%         if pred_xy_2sec(2) > xy_target_2s(2)
%             error_p2p_2s_temp = sqrt((pred(:,end-1) - xy_target_2s(1)).^2 + (pred(:,end) - xy_target_2s(2)).^2);
%             error_p2p_2s = min(error_p2p_2s_temp);
%         end
    end
    
    if ~isempty(pred_xy_3sec) && ~isempty(xy_target_3s)
        error_p2p_3s_temp = distance(pred_xy_3sec, xy_target_3s);
        error_p2p_3s = sqrt((pred_xy_3sec(1) - xy_target_3s(1))^2 + (pred_xy_3sec(2) - xy_target_3s(2))^2);
%         if pred_xy_3sec(2) > xy_target_3s(2)
%             error_p2p_3s_temp = sqrt((pred(:,end-1) - xy_target_3s(1)).^2 + (pred(:,end) - xy_target_3s(2)).^2);
%             error_p2p_3s = min(error_p2p_3s_temp);
%         end
    end
    
    if ~isempty(pred_xy_4sec) && ~isempty(xy_target_4s)
        error_p2p_4s_temp = distance(pred_xy_4sec, xy_target_4s);
        error_p2p_4s = sqrt((pred_xy_4sec(1) - xy_target_4s(1))^2 + (pred_xy_4sec(2) - xy_target_4s(2))^2);
%         if pred_xy_4sec(2) > xy_target_4s(2)
%             error_p2p_4s_temp = sqrt((pred(:,end-1) - xy_target_4s(1)).^2 + (pred(:,end) - xy_target_4s(2)).^2);
%             error_p2p_4s = min(error_p2p_4s_temp);
%         end
    end
    
    if ~isempty(pred_xy_5sec) && ~isempty(xy_target_5s)
        error_p2p_5s_temp = distance(pred_xy_5sec, xy_target_5s);
        error_p2p_5s = sqrt((pred_xy_5sec(1) - xy_target_5s(1))^2 + (pred_xy_5sec(2) - xy_target_5s(2))^2);
%         if pred_xy_5sec(2) > xy_target_5s(2)
%             error_p2p_5s_temp = sqrt((pred(:,end-1) - xy_target_5s(1)).^2 + (pred(:,end) - xy_target_5s(2)).^2);
%             error_p2p_5s = min(error_p2p_5s_temp);
%         end
    end
    
    error_P2P(i2, 1) = error_p2p_2s;
    error_P2P(i2, 2) = error_p2p_3s;
    error_P2P(i2, 3) = error_p2p_4s;
    error_P2P(i2, 4) = error_p2p_5s;

end
disp(error_Lat);


