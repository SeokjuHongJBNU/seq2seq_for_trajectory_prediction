%% 텍스트 파일에서 데이터를 가져옵니다.
% 다음 텍스트 파일에서 데이터를 가져오기 위한 스크립트:
%
%    C:\TrajectoryPrediction\LSTM\csv\rc1_220114.csv
%
% 선택한 다른 데이터나 텍스트 파일로 코드를 확장하려면 스크립트 대신 함수를 생성하십시오.

% MATLAB에서 다음 날짜에 자동 생성됨: 2022/01/14 11:01:08

%% 변수를 초기화합니다.
function output = vanilla_csv_load(csv_name)
filename = "C:\TrajectoryPrediction\LSTM\csv\LSTM_10Hz_ver02_RC" + csv_name;   %  rc1_220114.csv rc2_220114  rc3_220114  rc4_220114
delimiter = ',';

%% 각 텍스트 라인의 형식:
%   열1: double (%f)
%	열2: double (%f)
%   열3: double (%f)
%	열4: double (%f)
%   열5: double (%f)
%	열6: double (%f)
%   열7: double (%f)
%	열8: double (%f)
%   열9: double (%f)
%	열10: double (%f)
%   열11: double (%f)
%	열12: double (%f)
%   열13: double (%f)
%	열14: double (%f)
%   열15: double (%f)
%	열16: double (%f)
%   열17: double (%f)
%	열18: double (%f)
%   열19: double (%f)
%	열20: double (%f)
%   열21: double (%f)
%	열22: double (%f)
% 자세한 내용은 도움말 문서에서 TEXTSCAN을 참조하십시오.
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% 텍스트 파일을 엽니다.
fileID = fopen(filename,'r');

%% 형식에 따라 데이터 열을 읽습니다.
% 이 호출은 이 코드를 생성하는 데 사용되는 파일의 구조체를 기반으로 합니다. 다른 파일에 대한 오류가 발생하는 경우 가져오기 툴에서
% 코드를 다시 생성하십시오.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);

%% 텍스트 파일을 닫습니다.
fclose(fileID);

%% 가져올 수 없는 데이터에 대한 사후 처리 중입니다.
% 가져오기 과정에서 가져올 수 없는 데이터에 규칙이 적용되지 않았으므로 사후 처리 코드가 포함되지 않았습니다. 가져올 수 없는
% 데이터에 사용할 코드를 생성하려면 파일에서 가져올 수 없는 셀을 선택하고 스크립트를 다시 생성하십시오.

%% 출력 변수 만들기
rc1220114 = table(dataArray{1:end-1}, 'VariableNames', {'e01','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15','e16','e17','e18','e19','e20','e21'});

%% 임시 변수 지우기
clearvars filename delimiter formatSpec fileID dataArray ans;

output = rc1220114;