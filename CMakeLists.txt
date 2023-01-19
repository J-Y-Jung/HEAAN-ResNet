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

## add_executable(packing MPPacking.cpp) 
## target_link_libraries(packing PUBLIC HEaaN::HEaaN)

## add_executable(dsb DSB.cpp) 
## target_link_libraries(dsb PUBLIC HEaaN::HEaaN)

## add_executable(thread thread.cpp) 
## target_link_libraries(thread PUBLIC HEaaN::HEaaN)

## add_executable(test Inference_test.cpp) 
## target_link_libraries(test PUBLIC HEaaN::HEaaN)

add_executable(temptest temptest.cpp) 
target_link_libraries(temptest PUBLIC HEaaN::HEaaN)

## add_executable(apprelu AppReLUtest.cpp) 
## target_link_libraries(apprelu PUBLIC HEaaN::HEaaN)

## add_executable(conv Conv.cpp) 
## target_link_libraries(conv PUBLIC HEaaN::HEaaN)

add_executable(10000test 10000.cpp) 
target_link_libraries(10000test PUBLIC HEaaN::HEaaN)

add_executable(oddLazyBSGS oddLazyBSGStest.cpp) 
target_link_libraries(oddLazyBSGS PUBLIC HEaaN::HEaaN)

add_executable(inference Inference.cpp) 
target_link_libraries(inference PUBLIC HEaaN::HEaaN)

## add_executable(relubundle ReLUbundletest.cpp) 
## target_link_libraries(relubundle PUBLIC HEaaN::HEaaN)

## add_executable(rb RB+BN.cpp) 
## target_link_libraries(rb PUBLIC HEaaN::HEaaN)

