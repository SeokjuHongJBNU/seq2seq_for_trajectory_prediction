close all;  %clear; clc;

frame_sec = 1/2; % 25 frame per sec

fr_05sec = 0.5 / frame_sec;
fr_1sec = 1 / frame_sec;
fr_15sec = 1.5 / frame_sec;
fr_2sec = 2 / frame_sec;
fr_25sec = 2.5 / frame_sec;
fr_3sec = 3 / frame_sec;
fr_35sec = 3.5 / frame_sec;
fr_4sec = 4 / frame_sec;
fr_45sec = 4.5 / frame_sec;
fr_5sec = 5 / frame_sec;

hist_length = 3 / frame_sec;

%% 예측결과 csv 로딩
pred_csv = "result.csv";
result_csv_root = strcat("C:\TrajectoryPrediction\inD_LSTM\csv\", pred_csv);
pred = load(result_csv_root);

%% 원본데이터 로딩
ValidationData = "inD_LSTM_ver03_2Hz_group1_validation";  % inD_LSTM_ver02_validation  inD_LSTM_ver03_2Hz_group1_validation
ValidationSample_root = strcat("C:\TrajectoryPrediction\inD_LSTM\data\", ValidationData, ".mat");
load(ValidationSample_root, 'Test_X_stack', 'Test_Y_stack');

n4 = size(pred,2) * 0.5;
error_Lat = zeros(n4, 10);
error_P2P = zeros(n4, 10);

for i2 = 1 : n4

    x_trans_hist = Test_X_stack(:,i2, 1);
    y_trans_hist = Test_X_stack(:,i2, 2);
    x_trans_future_temp = Test_Y_stack(:,i2, 1);
    y_trans_future_temp = Test_Y_stack(:,i2, 2);
    x_trans_future = nonzeros(x_trans_future_temp); 
    y_trans_future = nonzeros(y_trans_future_temp);
    
    x_trans = [x_trans_hist; x_trans_future];
    y_trans = [y_trans_hist; y_trans_future];
    X_in_temp = [x_trans, y_trans];
    X_in = X_in_temp(1:2:end,:);

    % Lat Error
    pred_xy_05sec = pred(fr_05sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_1sec = pred(fr_1sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_15sec = pred(fr_15sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_2sec = pred(fr_2sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_25sec = pred(fr_25sec, (2*i2-1):(2*i2));
    pred_xy_3sec = pred(fr_3sec, (2*i2-1):(2*i2));
    pred_xy_35sec = pred(fr_35sec, (2*i2-1):(2*i2));
    pred_xy_4sec = pred(fr_4sec, (2*i2-1):(2*i2));
    pred_xy_45sec = pred(fr_45sec, (2*i2-1):(2*i2));
    pred_xy_5sec = pred(fr_5sec, (2*i2-1):(2*i2));
    
    if pred(fr_05sec, (2*i2)) > y_trans(end)
        idx_pred_05sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;        
        pred_xy_05sec_lat = pred(idx_pred_05sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_1sec, (2*i2)) > y_trans(end)
        idx_pred_1sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;    
        pred_xy_1sec_lat = pred(idx_pred_1sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_15sec, (2*i2)) > y_trans(end)
        idx_pred_15sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;   
        pred_xy_15sec_lat = pred(idx_pred_15sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_2sec, (2*i2)) > y_trans(end)
        idx_pred_2sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;        
        pred_xy_2sec_lat = pred(idx_pred_2sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_25sec, (2*i2)) > y_trans(end)
        idx_pred_25sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;        
        pred_xy_25sec_lat = pred(idx_pred_25sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_3sec, (2*i2)) > y_trans(end)
        idx_pred_3sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;        
        pred_xy_3sec_lat = pred(idx_pred_3sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_35sec, (2*i2)) > y_trans(end)
        idx_pred_35sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;        
        pred_xy_35sec_lat = pred(idx_pred_35sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_4sec, (2*i2)) > y_trans(end)
        idx_pred_4sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;    
        pred_xy_4sec_lat = pred(idx_pred_4sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_45sec, (2*i2)) > y_trans(end)
        idx_pred_45sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;    
        pred_xy_45sec_lat = pred(idx_pred_45sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_5sec, (2*i2)) > y_trans(end)
        idx_pred_5sec = find(pred(:,(2*i2)) >= y_trans(end), 1) - 1;   
        pred_xy_5sec_lat = pred(idx_pred_5sec, (2*i2-1):(2*i2));
    end
    
    pred_pos = [pred_xy_05sec; pred_xy_1sec; pred_xy_15sec; ...
        pred_xy_2sec; pred_xy_25sec; pred_xy_3sec_lat; pred_xy_35sec; ...
        pred_xy_4sec_lat; pred_xy_45sec_lat; pred_xy_5sec_lat];
    
%     pred_pos = [pred_xy_2sec; pred_xy_3sec_lat; pred_xy_4sec_lat; pred_xy_5sec_lat];

    figure;
    plot(x_trans_hist, y_trans_hist, 'b', 'linewidth', 2);  hold on; grid on; axis([50 200 -120 0]);
    plot(x_trans_future, y_trans_future, 'r');
    plot(pred(:,(2*i2-1)), pred(:,(2*i2)), 'k', 'linewidth', 1.1); % hold on; grid on; axis([30 90 30 90]);
    
    error_Lat = [];
    
    for j = 1:10
        [~, err_lat_sec, ~] = distance2curve([X_in(:,1), X_in(:,2)], [pred_pos(j,1) pred_pos(j,2)], 'spline');
        error_Lat(i2, j) = err_lat_sec;
    end

    %% Point-to-point 평가
    
    pred_xy_05sec = pred(fr_05sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_1sec = pred(fr_1sec, (2*i2-1):(2*i2));
    pred_xy_15sec = pred(fr_15sec, (2*i2-1):(2*i2));
    pred_xy_2sec = pred(fr_2sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_25sec = pred(fr_25sec, (2*i2-1):(2*i2));  % 20번째 sequence의 xy
    pred_xy_3sec = pred(fr_3sec, (2*i2-1):(2*i2));
    pred_xy_35sec = pred(fr_35sec, (2*i2-1):(2*i2));
    pred_xy_4sec = pred(fr_4sec, (2*i2-1):(2*i2));
    pred_xy_45sec = pred(fr_45sec, (2*i2-1):(2*i2));
    pred_xy_5sec = pred(fr_5sec, (2*i2-1):(2*i2));
    
    Pos_Current_GT_XY = [x_trans_hist(end), y_trans_hist(end)];
    plot(Pos_Current_GT_XY(1), Pos_Current_GT_XY(2), 'o', 'color', 'k');
    idx_current = find(y_trans >= Pos_Current_GT_XY(2), 1);

    idx_t_target_05s = hist_length + fr_05sec;
    idx_t_target_1s = hist_length + fr_1sec;
    idx_t_target_15s = hist_length + fr_15sec;
    idx_t_target_2s = hist_length + fr_2sec;
    idx_t_target_25s = hist_length + fr_25sec;
    idx_t_target_3s = hist_length + fr_3sec;
    idx_t_target_35s = hist_length + fr_35sec;
    idx_t_target_4s = hist_length + fr_4sec;
    idx_t_target_45s = hist_length + fr_45sec;
    idx_t_target_5s = hist_length + fr_5sec;
    
    if idx_t_target_05s > numel(x_trans)
        idx_t_target_05s = numel(x_trans);
    end
    
    if idx_t_target_1s > numel(x_trans)
        idx_t_target_1s = numel(x_trans);
    end
    
    if idx_t_target_15s > numel(x_trans)
        idx_t_target_15s = numel(x_trans);
    end
    
    if idx_t_target_2s > numel(x_trans)
        idx_t_target_2s = numel(x_trans);
    end
    
    if idx_t_target_25s > numel(x_trans)
        idx_t_target_25s = numel(x_trans);
    end
    
    if idx_t_target_3s > numel(x_trans)
        idx_t_target_3s = numel(x_trans);
    end
    
    if idx_t_target_35s > numel(x_trans)
        idx_t_target_35s = numel(x_trans);
    end
    
    if idx_t_target_4s > numel(x_trans)
        idx_t_target_4s = numel(x_trans);
    end
    
    if idx_t_target_45s > numel(x_trans)
        idx_t_target_45s = numel(x_trans);
    end
    
    if idx_t_target_5s > numel(x_trans)
        idx_t_target_5s = numel(x_trans);
    end
    
    xy_target_05s = [x_trans(idx_t_target_05s), y_trans(idx_t_target_05s)];
    xy_target_1s = [x_trans(idx_t_target_1s), y_trans(idx_t_target_1s)];
    xy_target_15s = [x_trans(idx_t_target_15s), y_trans(idx_t_target_15s)];
    xy_target_2s = [x_trans(idx_t_target_2s), y_trans(idx_t_target_2s)];
    xy_target_25s = [x_trans(idx_t_target_25s), y_trans(idx_t_target_25s)];
    xy_target_3s = [x_trans(idx_t_target_3s), y_trans(idx_t_target_3s)];
    xy_target_35s = [x_trans(idx_t_target_35s), y_trans(idx_t_target_35s)];
    xy_target_4s = [x_trans(idx_t_target_4s), y_trans(idx_t_target_4s)];
    xy_target_45s = [x_trans(idx_t_target_45s), y_trans(idx_t_target_45s)];
    xy_target_5s = [x_trans(idx_t_target_5s), y_trans(idx_t_target_5s)];
    plot(x_trans(idx_t_target_5s), y_trans(idx_t_target_5s), 'o');
    error_p2p_05s = NaN;
    error_p2p_1s = NaN;
    error_p2p_15s = NaN;
    error_p2p_2s = NaN;
    error_p2p_25s = NaN;
    error_p2p_3s = NaN;
    error_p2p_35s = NaN;
    error_p2p_4s = NaN;
    error_p2p_45s = NaN;
    error_p2p_5s = NaN;
    
    if ~isempty(pred_xy_05sec) && ~isempty(xy_target_05s)
        error_p2p_05s = sqrt((pred_xy_05sec(1) - xy_target_05s(1))^2 + (pred_xy_05sec(2) - xy_target_05s(2))^2);
    end
    
    if ~isempty(pred_xy_1sec) && ~isempty(xy_target_1s)
        error_p2p_1s = sqrt((pred_xy_1sec(1) - xy_target_1s(1))^2 + (pred_xy_1sec(2) - xy_target_1s(2))^2);
    end
    
    if ~isempty(pred_xy_15sec) && ~isempty(xy_target_15s)
        error_p2p_15s = sqrt((pred_xy_15sec(1) - xy_target_15s(1))^2 + (pred_xy_15sec(2) - xy_target_15s(2))^2);
    end
    
    if ~isempty(pred_xy_2sec) && ~isempty(xy_target_2s)
        error_p2p_2s = sqrt((pred_xy_2sec(1) - xy_target_2s(1))^2 + (pred_xy_2sec(2) - xy_target_2s(2))^2);
    end
    
    if ~isempty(pred_xy_25sec) && ~isempty(xy_target_25s)
        error_p2p_25s = sqrt((pred_xy_25sec(1) - xy_target_25s(1))^2 + (pred_xy_25sec(2) - xy_target_25s(2))^2);
    end
    
    if ~isempty(pred_xy_3sec) && ~isempty(xy_target_3s)
        error_p2p_3s = sqrt((pred_xy_3sec(1) - xy_target_3s(1))^2 + (pred_xy_3sec(2) - xy_target_3s(2))^2);
    end
    
    if ~isempty(pred_xy_35sec) && ~isempty(xy_target_35s)
        error_p2p_35s = sqrt((pred_xy_35sec(1) - xy_target_35s(1))^2 + (pred_xy_35sec(2) - xy_target_35s(2))^2);
    end
    
    if ~isempty(pred_xy_4sec) && ~isempty(xy_target_4s)
        error_p2p_4s = sqrt((pred_xy_4sec(1) - xy_target_4s(1))^2 + (pred_xy_4sec(2) - xy_target_4s(2))^2);
    end
    
    if ~isempty(pred_xy_45sec) && ~isempty(xy_target_45s)
        error_p2p_45s = sqrt((pred_xy_45sec(1) - xy_target_45s(1))^2 + (pred_xy_45sec(2) - xy_target_45s(2))^2);
    end
    
    if ~isempty(pred_xy_5sec) && ~isempty(xy_target_5s)
        error_p2p_5s = sqrt((pred_xy_5sec(1) - xy_target_5s(1))^2 + (pred_xy_5sec(2) - xy_target_5s(2))^2);
    end
    
    error_P2P(i2, 1) = error_p2p_05s;
    error_P2P(i2, 2) = error_p2p_1s;
    error_P2P(i2, 3) = error_p2p_15s;
    error_P2P(i2, 4) = error_p2p_2s;
    error_P2P(i2, 5) = error_p2p_25s;
    error_P2P(i2, 6) = error_p2p_3s;
    error_P2P(i2, 7) = error_p2p_35s;
    error_P2P(i2, 8) = error_p2p_4s;
    error_P2P(i2, 9) = error_p2p_45s;
    error_P2P(i2, 10) = error_p2p_5s;

end

