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

find_package(HEaaN REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(convtoolstest convtoolstest.cpp)
target_link_libraries(convtoolstest PUBLIC HEaaN::HEaaN)

add_executable(rotsumtest rotsumtest.cpp)
target_link_libraries(rotsumtest PUBLIC HEaaN::HEaaN)

add_executable(oddLazyBSGStest oddLazyBSGStest.cpp) 
target_link_libraries(oddLazyBSGStest PUBLIC HEaaN::HEaaN)

add_executable(AvgpoolFC64test AvgpoolFC64test.cpp) 
target_link_libraries(AvgpoolFC64test PUBLIC HEaaN::HEaaN)

