# HEAAN-ResNet project

how to add cpp files:
1. add .cpp into directory
2. add some lines to Cmakelists.txt (links HEAAN library to that .cpp)


how to build the project: 


cmake -B (원하는 디렉토리 이름)                  
cmake --build (원하는 디렉토리 이름)      
cd (원하는 디렉토리 이름)/bin              
./(실행 파일 이름)



Conv : 3 by 3 convolution만 구현 완료. 인자로 context, pack, eval, imgsize=32, gap, stride, ctxt, kernel_bundle(9개 커널 들어있는 묶음)을 받습니다.

MPPacking : gap이 벌어진 ctxt들(4개 혹은 16개)을 다시 묶어주는 역할. 인자로 context, pack, eval, imgsize=32, 모을 ctxt묶음을 받습니다.
