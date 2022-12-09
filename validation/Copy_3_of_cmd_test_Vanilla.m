close all;  %clear; clc;

cheby_t_intv = 100; % [Hz]

syms f(x, a1, a2, a3, a4, a5);
f = a1.*x.^4 + a2.*x.^3 + a3.*x.^2 + a4.*x + a5; % 4th Order Polynomial (Fitting)
nOrder_Cheby = 3; 
Pred_horizon = 5; % prediction horizon [s]
frame_sec = 1/25; % 25 frame per sec 
% cheby_t_intv = frame_sec; % [Hz]
error_MAE_dist_buff = [];
error_MAE_dist2_buff = [];
error_MAE_dist_cheby_buff = [];
error_MAE_dist2_cheby_buff = [];
ts = 0.04; %  25 Hz

fr_05sec = round(0.5 / frame_sec);
fr_1sec = round(1 / frame_sec);
fr_15sec = round(1.5 / frame_sec);

fr_2sec = round(2 / frame_sec);
fr_25sec = round(2.5 / frame_sec);
fr_3sec = round(3 / frame_sec);
fr_35sec = round(3.5 / frame_sec);
fr_4sec = round(4 / frame_sec);
fr_45sec = round(4.5 / frame_sec);
fr_5sec = round(5 / frame_sec);

% hist_length = 3 / frame_sec;

%% 예측결과 csv 로딩
if frame_sec == 1/2
    pred_csv = "result_2hz.csv";  % result_ver02b_25hz.csv   result_2hz.csv
    %% 원본데이터 로딩
    ValidationData = "inD_LSTM_ver03_2Hz_group1_validation_2sec";  % inD_LSTM_ver02_group1_validation  inD_LSTM_ver03_2Hz_group1_validation

elseif frame_sec == 1/25
    %  (1 hz_padding) result_25hz_padding_1hz / (2.5 hz) result_25hz_padding_2p5hz / (5 hz) result_25hz_padding_5hz_2
    pred_csv = "result_25hz_4feat.csv";  % result_ver02b_25hz.csv   result_2hz.csv  result_25hz_padding.csv
    % result_25hz_padding_5hz_2
    %% 원본데이터 로딩
    ValidationData = "inD_LSTM_ver02_group1_validation";  % inD_LSTM_ver02_group1_validation  inD_LSTM_ver03_2Hz_group1_validation
elseif frame_sec == 1/10
    pred_csv = "result_10hz.csv";  % result_ver02b_25hz.csv   result_2hz.csv
    %% 원본데이터 로딩
    ValidationData = "inD_LSTM_ver03_10Hz_group1_validation";
elseif frame_sec == 1/5
    pred_csv = "result_5hz.csv";
    ValidationData = "inD_LSTM_ver04_5Hz_validation_2sec";
end

result_csv_root = strcat("C:\TrajectoryPrediction\inD_LSTM\csv\", pred_csv);
pred = load(result_csv_root);
ValidationSample_root = strcat("C:\TrajectoryPrediction\inD_LSTM\data\", ValidationData, ".mat");
% load(ValidationSample_root, 'Test_X_stack', 'Test_Y_stack', 't_future_horizon_stack');
load(ValidationSample_root);
    
ValidationSample_GT = strcat("C:\TrajectoryPrediction\inD_LSTM\data\Validation_Sample_GT.mat");
load(ValidationSample_GT);

n4 = size(pred,2) * 0.5;
error_Lat = zeros(n4, 10);
error_P2P = zeros(n4, 10);
error_P2P_cheby = zeros(n4, 10);
error_Lat = [];

for i2 = 1 : n4
% for i2 = 10
    
    x_trans_hist = Test_X_stack(:,i2, 1);
    y_trans_hist = Test_X_stack(:,i2, 2);
    x_trans_future_temp = Test_Y_stack(:,i2, 1);
    y_trans_future_temp = Test_Y_stack(:,i2, 2);
    x_trans_future = nonzeros(x_trans_future_temp);
    y_trans_future = nonzeros(y_trans_future_temp);
    
    x_trans_25hz = nonzeros(X_stack_original(:, i2));  % Future GT 데이터 (x) 
    y_trans_25hz = nonzeros(Y_stack_original(:, i2));  % Future GT 데이터 (y) 
    
%     x_trans = nonzeros(Test_Y_stack(:, i2, 1));  % Future GT 데이터 (x) 
%     y_trans = nonzeros(Test_Y_stack(:, i2, 2));  % Future GT 데이터 (y) 
    
%     if numel(x_trans) > 125
%         x_trans = x_trans(1:125);
%         y_trans = y_trans(1:125);
%     end
    
    X_in_temp2 = [x_trans_25hz, y_trans_25hz];   % Future 전체 경로
    
    %% 2 Hz일 때 샘플 12개 간격 (0.5/0.04)
    if frame_sec == 1/2
        x_trans = x_trans_25hz(1:13:end);
        y_trans = y_trans_25hz(1:13:end);
    elseif frame_sec == 1/10
        x_trans = x_trans_25hz(1:25:end);
        y_trans = y_trans_25hz(1:25:end);
    elseif frame_sec == 1/5
        x_trans = x_trans_25hz(1:5:end);
        y_trans = y_trans_25hz(1:5:end);
    else
        x_trans = x_trans_25hz;
        y_trans = y_trans_25hz;
    end
    
%     x_trans = x_trans_future;
%     y_trans = y_trans_future;
%     X_in_temp = [x_trans_future, y_trans_future]; 
    X_in_temp = [x_trans, y_trans]; 
    
    X_in = X_in_temp(1:1:end,:);

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
    
    if pred(fr_05sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_05sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1) ;        
        pred_xy_05sec_lat = pred(idx_pred_05sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_1sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_1sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);    
        pred_xy_1sec_lat = pred(idx_pred_1sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_15sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_15sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);   
        pred_xy_15sec_lat = pred(idx_pred_15sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_2sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_2sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);        
        pred_xy_2sec_lat = pred(idx_pred_2sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_25sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_25sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);        
        pred_xy_25sec_lat = pred(idx_pred_25sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_3sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_3sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);        
        pred_xy_3sec_lat = pred(idx_pred_3sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_35sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_35sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);        
        pred_xy_35sec_lat = pred(idx_pred_35sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_4sec, (2*i2)) > y_trans(end)
        idx_pred_4sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);    
        pred_xy_4sec_lat = pred(idx_pred_4sec, (2*i2-1):(2*i2));
    end
    
    if pred(fr_45sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_45sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);    
        pred_xy_45sec_lat = pred(idx_pred_45sec, (2*i2-1):(2*i2));
    end
    
    pred_xy_5sec_lat = pred_xy_5sec;
    if pred(fr_5sec, (2*i2)) > X_in_temp2(end,2)
        idx_pred_5sec = find(pred(:,(2*i2)) >= y_trans_25hz(end), 1);   
        pred_xy_5sec_lat = pred(idx_pred_5sec, (2*i2-1):(2*i2));
    end
    
%     pred_pos = [pred_xy_05sec; pred_xy_1sec; pred_xy_15sec; ...
%         pred_xy_2sec; pred_xy_25sec; pred_xy_3sec; pred_xy_35sec; ...
%         pred_xy_4sec_lat; pred_xy_45sec_lat; pred_xy_5sec_lat];

    pred_pos_lat = [pred_xy_2sec; pred_xy_3sec; pred_xy_4sec; pred_xy_5sec];

    figure; 
    plot(x_trans_25hz, y_trans_25hz, 'k', 'linewidth', 3); hold on; grid on; axis([50 200 -120 0]);
    plot(x_trans_25hz(end), y_trans_25hz(end), '*');
    
    n_x_trans_25hz = numel(x_trans_25hz);
    if n_x_trans_25hz < 125
        plot(x_trans_25hz(1:n_x_trans_25hz), y_trans_25hz(1:n_x_trans_25hz), 'm', 'linewidth', 1); hold on; grid on; axis([50 200 -120 0]);
    else
        plot(x_trans_25hz(1:125), y_trans_25hz(1:125), 'm', 'linewidth', 1); hold on; grid on; axis([50 200 -120 0]);
    end
    
%     figure;
    plot(x_trans_hist, y_trans_hist, 'b', 'linewidth', 2);  hold on; grid on; axis([50 200 -120 0]);
%     plot(x_trans_future, y_trans_future, 'r', 'linewidth', 2);
    plot(pred(:,(2*i2-1)), pred(:,(2*i2)), 'g', 'linewidth', 1.1); % hold on; grid on; axis([30 90 30 90]);
%     plot(x_trans, y_trans, 'm', 'linewidth', 2); 
    for j = 1:4
        [~, err_lat_sec, ~] = distance2curve([X_in_temp2(:,1), X_in_temp2(:,2)], [pred_pos_lat(j,1) pred_pos_lat(j,2)], 'spline');
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
    
    idx_t_target_05s = fr_05sec;
    idx_t_target_1s = fr_1sec;
    idx_t_target_15s = fr_15sec;
    idx_t_target_2s = fr_2sec;
    idx_t_target_25s = fr_25sec;
    idx_t_target_3s = fr_3sec;
    idx_t_target_35s = fr_35sec;
    idx_t_target_4s = fr_4sec;
    idx_t_target_45s = fr_45sec;
    idx_t_target_5s = fr_5sec;
    
%     idx_t_target_05s =  round(0.5/ts);
%     idx_t_target_1s = round(1/ts);
%     idx_t_target_15s = round(1.5/ts);
%     idx_t_target_2s = round(2/ts);
%     idx_t_target_25s = round(2.5/ts);
%     idx_t_target_3s = round(3/ts);
%     idx_t_target_35s = round(3.5/ts);
%     idx_t_target_4s = round(4/ts);
%     idx_t_target_45s = round(4.5/ts);
%     idx_t_target_5s = round(5/ts);
    
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
%     plot(x_trans(idx_t_target_5s), y_trans(idx_t_target_5s), 'o');
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

    % 특정 시점 marker 표시하여 결과검토
    plot(xy_target_4s(1), xy_target_4s(2), 'sq', 'color', 'k');
    plot(xy_target_5s(1), xy_target_5s(2), 'sq', 'color', 'k');
    plot(pred_xy_4sec(1), pred_xy_4sec(2), 'o', 'color', 'k');
    plot(pred_xy_5sec(1), pred_xy_5sec(2), 'o', 'color', 'k');
    
    if isempty(xy_target_5s)
        a = 1;
    end

%% GT Chebyshev Transform

    pred_sample = pred(:,(2*i2-1):(2*i2));
    
    x = [-1 : 1/cheby_t_intv : 1];
    t_length_future = t_future_horizon_stack(j);
    time_future_snip = linspace(0, t_length_future, numel(pred_sample(:,1)))';
    
    poly_Pos_X_future = polyfit(time_future_snip, pred_sample(:,1), 4);
    poly_Pos_Y_future = polyfit(time_future_snip, pred_sample(:,2), 4);
    coeff_Pos_X_future_snip = cheby(f, nOrder_Cheby, poly_Pos_X_future, time_future_snip);
    coeff_Pos_Y_future_snip = cheby(f, nOrder_Cheby, poly_Pos_Y_future, time_future_snip);
    
    t_length_future = t_future_horizon_stack(j);
    x = linspace(-1, 1, 125);
    
    nGT = numel(x_trans);
    nHorizon = ceil(Pred_horizon/(1/25));
    GT_Pos_snippet = reshape(Test_Y_stack_GT(1:125, i2, 1:2), [125, 2]);
    
%     if nGT > nHorizon
%         GT_Pos_snippet(nHorizon+1:end, :) = [];
%     end
    
    if nOrder_Cheby == 3   % [ 1, x, 2*x^2 - 1, 4*x^3 - 3*x]
        c_x = coeff_Pos_X_future_snip;
        c_y = coeff_Pos_Y_future_snip;
        x_pred2 = c_x(:, 1)*1 + c_x(:, 2)*x + c_x(:, 3)*(2*x.^2 - 1) ...
            + c_x(:, 4)*(4*x.^3 - 3*x); % sum the first Five Chebyshev
        
        y_pred2 = c_y(:, 1)*1 + c_y(:, 2)*x + c_y(:, 3)*(2*x.^2 - 1) ...
            + c_y(:, 4)*(4*x.^3 - 3*x); % sum the first Five Chebyshev
        
    elseif nOrder_Cheby == 4
        c_x = coeff_Pos_X_future_snip;
        c_y = coeff_Pos_Y_future_snip;
        x_pred2 = c_x(:, 1)*1 + c_x(:, 2)*x + c_x(:, 3)*(2*x.^2 - 1) ...
            + c_x(:, 4)*(4*x.^3 - 3*x) + c_x(:, 5)*(8*x.^4 - 8*x.^2 + 1); % sum the first Five Chebyshev

        y_pred2 = c_y(:, 1)*1 + c_y(:, 2)*x + c_y(:, 3)*(2*x.^2 - 1) ...
            + c_y(:, 4)*(4*x.^3 - 3*x) + c_y(:, 5)*(8*x.^4 - 8*x.^2 + 1); % sum the first Five Chebyshev
    end
    
    pred2 = [x_pred2', y_pred2'];
    
    if size(X_in_temp2, 1) > 125
        X_in_temp2(126 : end, :) = [];
    end
    
    if size(pred2(:,1), 1) > size(X_in_temp2(:,1), 1)
        pred2(size(X_in_temp2(:,1), 1)+1:end, :) = [];
    end
    
    idx_3sec = round(3/0.04);
    idx_35sec = round(3.5/0.04);
    idx_4sec = round(4/0.04);
    idx_45sec = round(4.5/0.04);
    idx_5sec = round(5/0.04);
    
    % Chebyshev 변환 결과의 평가 시점
    if idx_3sec > size(pred2, 1)
        idx_3sec = size(pred2, 1);
    end
    
    if idx_35sec > size(pred2, 1)
        idx_35sec = size(pred2, 1);
    end
    
    if idx_4sec > size(pred2, 1)
        idx_4sec = size(pred2, 1);
    end
    
    if idx_45sec > size(pred2, 1)
        idx_45sec = size(pred2, 1);
    end
    
    if idx_5sec > size(pred2, 1)
        idx_5sec = size(pred2, 1);
    end
    
    pred_xy_3sec = pred2(idx_3sec, :);
    pred_xy_35sec = pred2(idx_35sec, :);
    pred_xy_4sec = pred2(idx_4sec, :);
    pred_xy_45sec = pred2(idx_45sec, :);
    pred_xy_5sec = pred2(idx_5sec, :);

    error_p2p_3s = NaN;
    error_p2p_35s = NaN;
    error_p2p_4s = NaN;
    error_p2p_45s = NaN;
    error_p2p_5s = NaN;
    
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
    
    error_P2P_cheby(i2, 6) = error_p2p_3s;
    error_P2P_cheby(i2, 7) = error_p2p_35s;
    error_P2P_cheby(i2, 8) = error_p2p_4s;
    error_P2P_cheby(i2, 9) = error_p2p_45s;
    error_P2P_cheby(i2, 10) = error_p2p_5s;
    
    %% MAE 결과 검토 (오리지널 gt에 MAE 계산)
    n_gt = size(X_in_temp2, 1);
    error_p2p_total_points = sqrt((pred(1:n_gt,(2*i2-1)) - X_in_temp2(:,1)).^2 + (pred(1:n_gt,(2*i2)) - X_in_temp2(:,2)).^2);
    error_MAE_dist = sum(abs(error_p2p_total_points)) / numel(error_p2p_total_points);
    error_MAE_dist_buff = [error_MAE_dist_buff; error_MAE_dist];
    
    start_sec = 3; % 특정시점(2초) 이후부터 MAE 평가
    idx_3sec = round(start_sec/ts);
    error_p2p_after3sec = sqrt((pred2(idx_3sec:end,1) - X_in_temp2(idx_3sec:end,1)).^2 + ...
        (pred2(idx_3sec:end,2) - X_in_temp2(idx_3sec:end,2)).^2);
    error_MAE_dist2_cheby = sum(abs(error_p2p_after3sec)) / numel(error_p2p_after3sec);
    error_MAE_dist2_cheby_buff = [error_MAE_dist2_cheby_buff; error_MAE_dist2_cheby];
    
    %% MAE 결과: Chebyshev 변환 시 
    % X_in_temp2 (Future 전체 경로 - 25 Hz)
    error_p2p_total_points = sqrt((pred2(:,1) - X_in_temp2(:,1)).^2 + (pred2(:,2) - X_in_temp2(:,2)).^2);
    error_MAE_dist_cheby = sum(abs(error_p2p_total_points)) / numel(error_p2p_total_points);
    error_MAE_dist_cheby_buff = [error_MAE_dist_cheby_buff; error_MAE_dist_cheby];
    
    start_sec = 3; % 특정시점(2초) 이후부터 MAE 평가
    idx_3sec = round(start_sec/ts);
    error_p2p_after3sec = sqrt((pred2(idx_3sec:end,1) - X_in_temp2(idx_3sec:end,1)).^2 + ...
        (pred2(idx_3sec:end,2) - X_in_temp2(idx_3sec:end,2)).^2);
    error_MAE_dist2_cheby = sum(abs(error_p2p_after3sec)) / numel(error_p2p_after3sec);
    error_MAE_dist2_cheby_buff = [error_MAE_dist2_cheby_buff; error_MAE_dist2_cheby];
    
%     plot(pred2(:,1),pred2(:,2), 'g');
    
end

