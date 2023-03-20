# HEAAN-ResNet project

how to add cpp files:
1. add .cpp into directory
2. add some lines to Cmakelists.txt (links HEAAN library to that .cpp)


how to build the project: 


cmake -B (directory)                  
cmake --build (directory)      
cd (directory)/bin              
./(filename)

## ideas
1. Packing :  amortized time을 위해 하나의 이미지의 3ch을 4ch로 하나의 ctxt에 packing하지 않고, 3개의 ctxt에 packing하여 slot을 더 밀도 있게 사용함. conv에 불필요한 rot & sum을 없앨 수 있음. 4ch일 때보다 rotation 2번 아낄 수 있음
2. MPP : DSB에서 MPP 순서를 바꾸면 MPP 횟수는 더 늘어나지만, 연산량 최적화 가능해 보임

<hr/>

#### Conv 

Conv : 3 by 3 convolution만 구현 완료. 인자로 context, pack, eval, imgsize=32, gap, stride, input channel, output channel, ctxt_bundle(input channel개 만큼), kernel_bundle을 받습니다.

MPPacking : gap이 벌어진 ctxt들(4개 혹은 16개)을 다시 묶어주는 역할. 인자로 context, pack, eval, imgsize=32, 모을 ctxt묶음을 받습니다.

<hr/>

#### DSB (Down Sampling Block) 

DSB.cpp 구동 확인 완료.

함수 이름 : DSB

인자 : context, pack, eval, ctxt_bundle, kernel_bundle(처음 3 by 3 kernel), kernel_bundle2(두번째 3 by 3 kernel), kernel_residual_bundle(residual flow의 bundle).

수정 필요 : MPP 순서 변경으로 최적화 예정(설명 추가 예정)

<hr/>

#### RB (Residual Block) 

완성 (설명 추가 예정)

<hr/>

#### convtools 

vector<vector<double>>형태의 커널 값을 원하는 convolution의 ptxt로 패킹해주는 함수.

vector<double>이 하나의 "output channel"에 해당하는 값이므로

1. SIMD inference of multiple images의 경우 vector<vector<double>>은 같은 vector<double>의 반복

2. (original 논문 버전의) single image inference의 경우 서로 다른 output channel의 vector<double>을 벡터로 묶어서 인자로 받음

[2. 의 경우 conv이후 MPpacking 시 적절한 mask와 추가적인 rotation을 사용하여 각 채널의 결과를 분리해준 후 모아야 함]

TODO: heaan message 저장 기능을 추가하여 precomputing 해놓기

<hr/>

#### rotsum 

RotSum2Idx : 원하는 간격(power of 2)으로 원하는 원소 갯수(power of 2)만큼 rotation-sum 한 후 올바른 결과값이 지정한 Index에 해당하는 슬롯에 오도록 하는 함수. FC, Conv후 결과물을 합칠 때 사용 가능.

다른 슬롯에는 trash값이 올 가능성이 있으므로 사용 시 주의. 일반적으로 log(원소 갯수)번 rotation이 일어나지만 Index가 원하는 간격의 배수가 아닌 경우 추가적으로 1번 rotation이 일어남.

<hr/>

#### Avgpool + FC64 

Avgpool : 8*8개의 각 픽셀 값들을 모두 더함 (1/64 는 곱하지 않음).
  
각 (32*32) 블록의 (0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67, 96, 97, 98, 99) 인덱스에만 올바른 값이 저장되어있음.
  
FC64 : aaa

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
  폴더째로 넣으려 하면 안되는 듯 합니다... 번거롭지만 잘 안되신다면 파일 하나씩 넣어보세요.
  
  > sudo docker cp /home/(WinSCP 폴더이름)/파일test.cpp dockerID:/app/examples
  
  > sudo docker exec -it  dockerID /bin/bash

  > cmake -B build
  
  > cmake --build build
  
  > cd build/bin -> ./실행파일이름
  
파일 업데이트 될 때마다 4번부터 반복 실행



### kernelEncode, imageEncode
실행파일이 있는 위치에서 각각 kernel, image 폴더가 있으면 됨
