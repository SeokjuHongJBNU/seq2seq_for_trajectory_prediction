%% �ؽ�Ʈ ���Ͽ��� �����͸� �����ɴϴ�.
% ���� �ؽ�Ʈ ���Ͽ��� �����͸� �������� ���� ��ũ��Ʈ:
%
%    C:\TrajectoryPrediction\LSTM\csv\rc1_220114.csv
%
% ������ �ٸ� �����ͳ� �ؽ�Ʈ ���Ϸ� �ڵ带 Ȯ���Ϸ��� ��ũ��Ʈ ��� �Լ��� �����Ͻʽÿ�.

% MATLAB���� ���� ��¥�� �ڵ� ������: 2022/01/14 11:01:08

%% ������ �ʱ�ȭ�մϴ�.
function output = vanilla_csv_load(csv_name)
filename = "C:\TrajectoryPrediction\LSTM\csv\LSTM_10Hz_ver02_RC" + csv_name;   %  rc1_220114.csv rc2_220114  rc3_220114  rc4_220114
delimiter = ',';

%% �� �ؽ�Ʈ ������ ����:
%   ��1: double (%f)
%	��2: double (%f)
%   ��3: double (%f)
%	��4: double (%f)
%   ��5: double (%f)
%	��6: double (%f)
%   ��7: double (%f)
%	��8: double (%f)
%   ��9: double (%f)
%	��10: double (%f)
%   ��11: double (%f)
%	��12: double (%f)
%   ��13: double (%f)
%	��14: double (%f)
%   ��15: double (%f)
%	��16: double (%f)
%   ��17: double (%f)
%	��18: double (%f)
%   ��19: double (%f)
%	��20: double (%f)
%   ��21: double (%f)
%	��22: double (%f)
% �ڼ��� ������ ���� �������� TEXTSCAN�� �����Ͻʽÿ�.
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% �ؽ�Ʈ ������ ���ϴ�.
fileID = fopen(filename,'r');

%% ���Ŀ� ���� ������ ���� �н��ϴ�.
% �� ȣ���� �� �ڵ带 �����ϴ� �� ���Ǵ� ������ ����ü�� ������� �մϴ�. �ٸ� ���Ͽ� ���� ������ �߻��ϴ� ��� �������� ������
% �ڵ带 �ٽ� �����Ͻʽÿ�.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);

%% �ؽ�Ʈ ������ �ݽ��ϴ�.
fclose(fileID);

%% ������ �� ���� �����Ϳ� ���� ���� ó�� ���Դϴ�.
% �������� �������� ������ �� ���� �����Ϳ� ��Ģ�� ������� �ʾ����Ƿ� ���� ó�� �ڵ尡 ���Ե��� �ʾҽ��ϴ�. ������ �� ����
% �����Ϳ� ����� �ڵ带 �����Ϸ��� ���Ͽ��� ������ �� ���� ���� �����ϰ� ��ũ��Ʈ�� �ٽ� �����Ͻʽÿ�.

%% ��� ���� �����
rc1220114 = table(dataArray{1:end-1}, 'VariableNames', {'e01','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15','e16','e17','e18','e19','e20','e21'});

%% �ӽ� ���� �����
clearvars filename delimiter formatSpec fileID dataArray ans;

output = rc1220114;