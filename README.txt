
# LSTM seq2seq를 이용해 차량 경로 예측하기




## 코드를 원활하게 실행하기 위해서는 /Desktop/My_LSTM_example/My_LSTM_example 다음과 같은 경로로 되어있어야 합니다. (폴더명 변경해야 합니다.)
## 실행방법
'train_run.py' 파일에 iw, ow, s, num_feature 변수를 사용자가 원하는 값으로 변경하고, 'test_run.py' 파일에도 동일하게 적용시켜줍니다.

그 후 'train_run.py'로 학습하고, 'test_run.py'로 결과를 확인하면 됩니다.

## Requirements 
- Python 3+
- PyTorch
- numpy
- pandas
- matplotlib


## 1 코드의 목적
이 코드는 LSTM seq2seq를 이용해 차량의 경로를 학습하고 테스트하고, plot을 통해 예측 모델이 얼만큼 잘 작동하는지 확인할 수 있습니다.
이전 시점의 데이터를 input으로 받아 미래를 예측하는 것입니다.



## LSTM encoder-decoder
LSTM 인코더-디코더(seq2seq)는 크게 두개의 RNN(LSTM) 모형을 합친 형태인데요, 이것의 장점으로는 인풋 시퀀스와 아웃풋 시퀀스의 크기가 달라도 작동을 할 수 있다는 점입니다.


## 2 Preparing Dataset
데이터는 차량의 X, Y 좌표, Velocity(이하 Vel), Yaw rate(차량이 향하는 각도 변화율, 이하 Y.R) 정보를 포함합니다. 
input data에서 X, Y 좌표는 필수로 포함되어야 합니다. 
따라서 만들 수 있는 feature 조합 수는 feature = 2(X, Y), feature = 3(X, Y, Vel or Y.R) 그리고 feature = 4(X, Y, Vel and Y.R) 총 4개의 가능한 input feature 개수를 만들 수 있다. 
output은 차량의 X, Y좌표이다. 실험 데이터로는 120대의 차량 궤적을 train data, 10대의 차량 궤적을 test data로 설정했습니다.

우리는 데이터를  `generate_dataset.py`과 'build_data.py' 에서 만듭니다.
'generate_dataset.py'에서는 windowed_data를 만들어주는 함수인데요, 데이터를 원하는 input 시퀀스 길이, output 시퀀스 길이로 다르게 생성할 수 있습니다. 
또한, 하나의 경로 데이터에 대해 여러개의 데이터를 생성할 수 있어 데이터의 양을 늘릴 수 있습니다.

'build_data.py'에서는 사용자가 원하는 input_window, output_window, stride, num_feature를 입력했을 때,
이때 가지고 있는 데이터에서 원하는 피쳐를 불러들여와 결측치 처리 등 과정을 거치고 사용자가 입력한 조합에 맞게 데이터를 생성해줍니다.
만약 피쳐가 3개라면, 가능한 경우의 수가 2개(vel, Y.R)이므로 terminal에 0(vel) 또는 1(Y.R)을 입력해야 합니다.
또한 normalization을 위한 max값, min값도 위 함수에서 만들어 줍니다.

## 3 'train_run.py'
'build_data.py'에서 생성한 데이터를 normalization 시키고 훈련시킵니다.
그리고 loss값과 val_loss 값을 확인할 수 있습니다.

## 4. 'test_run.py'
훈련된 모델을 test 합니다.
X_test가 모델에 input되어 Y_predict를 만들어 냅니다. 이 때 Y_test와 비교할 수 있는 애니메이션 plot을 확인할 수 있습니다.





