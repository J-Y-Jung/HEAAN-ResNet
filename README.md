# HEAAN-ResNet project

how to add cpp files:
1. add .cpp into directory
2. add some lines to Cmakelists.txt (links HEAAN library to that .cpp)


how to build the project: 


cmake -B (원하는 디렉토리 이름)                  
cmake --build (원하는 디렉토리 이름)      
cd (원하는 디렉토리 이름)/bin              
./(실행 파일 이름)

## (사소한) ideas
1. Packing : (용동) amortized time을 위해 하나의 이미지의 3ch을 4ch로 하나의 ctxt에 packing하지 않고, 3개의 ctxt에 packing하여 slot을 더 밀도 있게 사용함. conv에 불필요한 rot & sum을 없앨 수 있음.

2. Avgpool + FC64 : (준영)
3. 

## CryptoLab 미팅 때 확인해야 할 것들
1. Thread 사용량 (BTS 몇 thread로 구동?, multi-thread 할당하는 법?, 등등)
2. Bootstrapping 조작 가능 범위?
3. BTS 구현 방식? (SOTA?)
4. 만약 정식 HEaaN을 사용할 필요가 있다면, 문법이 얼마나 달라지는지?

## Issues

1. ctxt * msg 가 너무 오래 걸림. 인코딩과 리스케일링이 시간을 상당히 많이 잡아먹음. 커널 인코딩은 전부 전처리 단계에서 하는 것이 맞는 것 같음. HEaaN::EnDecoder를 사용하여 전부 ptxt(인코딩된 형태)로 바꾼 후 ctxt * ptxt 로 계산하되, 가능한 한 multWithoutRescale로 연산 후 나중에 한꺼번에 rescale해야 함.

(M1 맥북 기준: ctxt * msg : <300ms, ctxt * ptxt : <100ms, multWithoutRescale : <10ms)

현재 convtools, AvgpoolFC64 등을 전부 ctxt * ptxt 연산, hoisted rescaling을 사용하도록 변경 중 (준영)

11/16 : Conv에서 ctxt * ptxt 연산으로 수정 완료. hoisting 구현 완료. 여전히 전체 시간에서 convolution이 차지하는 비율이 커서 불만족스럽지만 일단 구현해보기로 결정. (용동)

AvgpoolFC64 ctxt * ptxt 연산으로 수정 완료. FC64의 경우 현재 구현이 sum(rot(mult(ct, pt), idx)) 형태로 되어 있어서 hoisting 아쉽게도 사용 불가. 

convtools의 함수들을 ptxt를 출력하도록 변경. 함수명도 바뀜 (Weight2Msg -> weightToPtxt). 입력받는 변수들도 변경이 있음 (미리 정의된 HEaaN::EnDecoder를 넣어줘야함)

<hr/>

#### Conv (용동)

Conv : 3 by 3 convolution만 구현 완료. 인자로 context, pack, eval, imgsize=32, gap, stride, input channel, output channel, ctxt_bundle(input channel개 만큼), kernel_bundle을 받습니다.

MPPacking : gap이 벌어진 ctxt들(4개 혹은 16개)을 다시 묶어주는 역할. 인자로 context, pack, eval, imgsize=32, 모을 ctxt묶음을 받습니다.

<hr/>

#### DSB (Down Sampling Block) (용동)

DSB.cpp 구동 확인 완료.(현재 구동 불가 상태. 빠른 시일 내 업데이트 예정.)

함수 이름 : DSB

인자 : context, pack, eval, ctxt_bundle (MPP를 함으로서 4개의 ctxt(32개 이미지)를 동시에 처리함.), kernel_bundle(처음 3 by 3 kernel), kernel_bundle2(두번째 3 by 3 kernel), kernel_residual_bundle(residual flow의 bundle).

수정 필요 : 16채널을 output으로 뱉는 것이 아니라 1채널 뱉음.


<hr/>

#### RB (Residual Block) (용동)

11/11 이전 업데이트 예정. DSB 먼저 업데이트 예정.

<hr/>

#### convtools (준영)

vector<vector<double>>형태의 커널 값을 원하는 convolution의 ptxt로 패킹해주는 함수.

vector<double>이 하나의 "output channel"에 해당하는 값이므로

1. SIMD inference of multiple images의 경우 vector<vector<double>>은 같은 vector<double>의 반복

2. (original 논문 버전의) single image inference의 경우 서로 다른 output channel의 vector<double>을 벡터로 묶어서 인자로 받음

[2. 의 경우 conv이후 MPpacking 시 적절한 mask와 추가적인 rotation을 사용하여 각 채널의 결과를 분리해준 후 모아야 함]

TODO: heaan message 저장 기능을 추가하여 I/O랑 통합?

<hr/>

#### rotsum (준영)

RotSum2Idx : 원하는 간격(power of 2)으로 원하는 원소 갯수(power of 2)만큼 rotation-sum 한 후 올바른 결과값이 지정한 Index에 해당하는 슬롯에 오도록 하는 함수. FC, Conv후 결과물을 합칠 때 사용 가능.

다른 슬롯에는 trash값이 올 가능성이 있으므로 사용 시 주의. 일반적으로 log(원소 갯수)번 rotation이 일어나지만 Index가 원하는 간격의 배수가 아닌 경우 추가적으로 1번 rotation이 일어남.

<hr/>

#### Avgpool + FC64 (준영)

Avgpool : 8*8개의 각 픽셀 값들을 모두 더함 (1/64 는 곱하지 않음).
  
각 (32*32) 블록의 (0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99) 인덱스에만 올바른 값이 저장되어있음.
  
FC64 : (bias 없음) 64*10 행렬의 각 행을 적절한 형태의 message[각 (32*32) 블록의 (0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99) 인덱스에만 값이 있고 나머지는 0인]로 변환 후 avgpool 완료한 ctxt와 각각 곱한 후, 적절히 rotate-sum 한 후 하나의 ctxt로 합쳐 주는 기능.
  
구현 상 이슈로 결과로 나온 ctxt의 (0, 8, 16, 24, 256, 264, 272, 280, 512, 520)에 해당하는 인덱스에만 올바른 값이 들어있음.
  
나머지 인덱스에 trash값이 들어있으므로, 실제 사용 시에는 한번 masking해줘야 하지만, depth를 잡아먹으므로... 생략함
  
bias는 적절한 모양으로 변형 뒤 더해 주면 됨

<hr/>

#### ReLU

CKKS 최신 논문에서 소개된 주어진 (odd) polynomial f 와 ctxt 에 대해 f(ctxt)를 evaluate 하는 lazy BSGS 알고리즘 구현,

이를 응용해 ReLU 함수를 ||ReLU - f||_[-1, 1] <2^{-13} 인 근사 다항식 f로 근사해 evaluate 하는 알고리즘 구현 완료.

ResNet input들의 절대값 upper bound B에 대해 (B=40), f(x) 대신 B*f(x/B) 로 evaluate 해야함

그 외에 다른 구성요소와 결합할 때 고려해야 하는 부분은 추후에 생각날 때 마다 기입함

void ApproximateReLU(context, eval, ctxt_in, ctxt_out) 꼴로 정의되어 있음

<hr/>

#### Weight2Vec 


1개의 열로 구성된 weight를 cpp에서 vector화 시키는 함수.

3d vector로 구성.

e.g) number of output_channel = 16 , number of input_channel = 3 , kernel_size = 3

###### < output1{ [input_1],[inpu_2],[input_3] }, output2{ [input_1],[input_2],[input_3] } ,  ... ,output16{ [input_1],[input_2],[input_3]} > 


#### Summand2Vec
summand part의 data를 vector화 시키는 함수.

<hr/>
  
 
#### Parameters (Filters and batch normalizations blended)

filter 와 batch normalization 을 통합한 파라미터들.
  

block 0 은 3채널 -> 16채널 convolution 레이어 하나.
  
block 1~3 은 regular blocks.
  
block 4 는 downsampling block 이고 block 5~6 은 regular blocks.
  
block 7 은 downsampling block 이고 block 8~9 는 regular blocks.
  

conv 레이어를 통과시킬 때, multiplicands 를 filter 로 보고 summands 를 bias 로 보면 됨.
  
그러면 filter + batch normalization 을 한꺼번에 한 결과가 나옴.

<hr/>
  
#### Server tutorial
  
  필요 : Putty , WinSCP
  
  
  1.Putty , WinSCP 접속
  
  2.WinSCP에 heaan-native-docker.tar.gz파일 올리기 
  
  3.Putty에 docker container image load 하기
  
  4.WinSCP 폴더에 파일 올리기(hpp,cpp파일). (CMakeList.txt도 file에 따라서 update시켜줘야함)
  
  5.Putty 명렁어
  
  > sudo docker cp /home/(WinSCP 폴더이름)/파일.hpp dockerID:/app/examples 
  폴더째로 넣으려 하면 안되는 듯 합니다... 번거롭지만 잘 안되신다면 파일 하나씩 넣어보세요.(용동)
  
  > sudo docker cp /home/(WinSCP 폴더이름)/파일test.cpp dockerID:/app/examples
  
  > sudo docker exec -it  dockerID /bin/bash

  > cmake -B build
  
  > cmake --build build
  
  > cd build/bin -> ./실행파일이름
  
파일 업데이트 될 때마다 4번부터 반복 실행
