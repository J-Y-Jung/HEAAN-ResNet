# HEAAN-ResNet project

how to add cpp files:
1. add .cpp into directory
2. add some lines to Cmakelists.txt (links HEAAN library to that .cpp)


how to build the project: 


cmake -B (원하는 디렉토리 이름)                  
cmake --build (원하는 디렉토리 이름)      
cd (원하는 디렉토리 이름)/bin              
./(실행 파일 이름)

---------------------------------------------------------------------------------------------------

#### Conv

Conv : 3 by 3 convolution만 구현 완료. 인자로 context, pack, eval, imgsize=32, gap, stride, ctxt, kernel_bundle(9개 커널 들어있는 묶음)을 받습니다.

MPPacking : gap이 벌어진 ctxt들(4개 혹은 16개)을 다시 묶어주는 역할. 인자로 context, pack, eval, imgsize=32, 모을 ctxt묶음을 받습니다.

---------------------------------------------------------------------------------------------------

#### DSB (Down Sampling Block)

DSB.cpp 구동 확인 완료. .hpp로 추후에 가까운 시일 내에 업데이트 예정

---------------------------------------------------------------------------------------------------

#### ReLU

CKKS 최신 논문에서 소개된 주어진 (odd) polynomial f 와 ctxt 에 대해 f(ctxt)를 evaluate 하는 lazy BSGS 알고리즘 구현,

이를 응용해 ReLU 함수를 ||ReLU - f||_[-1, 1] <2^{-13} 인 근사 다항식 f로 근사해 evaluate 하는 알고리즘 구현 완료.

ResNet input들의 절대값 upper bound B에 대해 (B=40), f(x) 대신 B*f(x/B) 로 evaluate 해야함

그 외에 다른 구성요소와 결합할 때 고려해야 하는 부분은 추후에 생각날 때 마다 기입함

void ApproximateReLU(context, eval, ctxt_in, ctxt_out) 꼴로 정의되어 있음
