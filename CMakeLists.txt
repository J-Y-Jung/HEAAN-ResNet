################################################################################
##                                                                            ##
## Copyright (C) 2021-2022 Crypto Lab Inc.                                    ##
##                                                                            ##
## - This file is part of HEaaN homomorphic encryption library.               ##
## - HEaaN cannot be copied and/or distributed without the express permission ##
##  of Crypto Lab Inc.                                                        ##
##                                                                            ##
################################################################################

project(HEaaN_examples LANGUAGES CXX)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(HEaaN REQUIRED)

add_executable(nolazyBSGS nolazyBSGS.cpp) 
target_link_libraries(nolazyBSGS PUBLIC HEaaN::HEaaN)

add_executable(inference Inference40.cpp) 
target_link_libraries(inference PUBLIC HEaaN::HEaaN)
