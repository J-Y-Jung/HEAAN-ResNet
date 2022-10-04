# HEAAN-ResNet project

how to add cpp files:
1. add .cpp into directory
2. add some lines to Cmakelists.txt (links HEAAN library to that .cpp)


how to build the project
cmake -B cnn                   // makes directory named cnn
cmake --build cnn              //and build onto that directory
cd cnn/bin                     //executable binary files can be found on here
./convtools
