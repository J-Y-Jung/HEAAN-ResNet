////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "HEaaN/device/CudaTools.hpp"
#include "HEaaN/device/Device.hpp"
#include <chrono>
#include <iostream>
#include <string>

namespace HEaaN {
class HEaaNTimer {
public:
    HEaaNTimer(bool is_device = true)
        : timing_device_functions_{CudaTools::isAvailable() && is_device} {}
    void start(std::string name) {
        name_ = std::move(name);
        if (timing_device_functions_)
            CudaTools::cudaDeviceSynchronize();
        beg_ = std::chrono::high_resolution_clock::now();
    }
    void end() {
        if (timing_device_functions_)
            CudaTools::cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto dur =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - beg_)
                .count();

        std::cout << name_ << "\033[0m"
                  << " time = ";

        if (dur < 1000)
            std::cout << dur << " ms" << std::endl;
        else {
            std::cout.precision(3);
            std::cout << (double)dur / 1000.0 << " s" << std::endl;
        }
    }

private:
    std::string name_;
    bool timing_device_functions_{false};
    std::chrono::time_point<std::chrono::high_resolution_clock> beg_;

}; // class HEaaNTimer

} // namespace HEaaN
