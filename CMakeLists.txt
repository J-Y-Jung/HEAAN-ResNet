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

add_executable(convtoolstest convtoolstest.cpp)
target_link_libraries(convtoolstest PUBLIC HEaaN::HEaaN)