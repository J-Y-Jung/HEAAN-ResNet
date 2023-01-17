////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <optional>
#include <random>

#include "HEaaN/heaan.hpp"

std::default_random_engine gen{std::random_device()()};

inline long double randNum() {
    std::uniform_real_distribution<long double> dist(-1.0L, 1.0L);
    return dist(gen);
}

void fillRandomComplex(HEaaN::Message &msg) {
    for (size_t i = 0; i < msg.getSize(); ++i) {
        msg[i].real(randNum());
        msg[i].imag(randNum());
    }
}

void fillRandomReal(HEaaN::Message &msg, std::optional<size_t> num) {
    size_t length = num.has_value() ? num.value() : msg.getSize();
    size_t idx = 0;
    for (; idx < length; ++idx) {
        msg[idx].real(randNum());
        msg[idx].imag(0.0);
    }
    // If num is less than the size of msg,
    // all remaining slots are zero.
    for (; idx < msg.getSize(); ++idx) {
        msg[idx].real(0.0);
        msg[idx].imag(0.0);
    }
}

void printMessage(const HEaaN::Message &msg, bool is_complex = true,
                  size_t start_num = 16, size_t end_num = 2) {

    std::cout.precision(7);

    const size_t msg_size = msg.getSize();

    std::cout << "[ ";
    for (size_t i = 0; i < start_num; ++i) {
        if (is_complex)
            std::cout << msg[i] << ", ";
        else
            std::cout << msg[i].real() << ", ";
    }
    std::cout << "..., ";
    for (size_t i = end_num; i > 1; --i) {
        if (is_complex)
            std::cout << msg[msg_size - i] << ", ";
        else
            std::cout << msg[msg_size - i].real() << ", ";
    }
    if (is_complex)
        std::cout << msg[msg_size - 1] << " ]" << std::endl;
    else
        std::cout << msg[msg_size - 1].real() << " ]" << std::endl;
}

std::string presetNamer(const HEaaN::ParameterPreset preset) {
    switch (preset) {
    case HEaaN::ParameterPreset::FVa:
        return "FVa";
    case HEaaN::ParameterPreset::FVb:
        return "FVb";
    case HEaaN::ParameterPreset::FGa:
        return "FGa";
    case HEaaN::ParameterPreset::FGb:
        return "FGb";
    case HEaaN::ParameterPreset::FTa:
        return "FTa";
    case HEaaN::ParameterPreset::FTb:
        return "FTb";
    case HEaaN::ParameterPreset::ST19:
        return "ST19";
    case HEaaN::ParameterPreset::ST14:
        return "ST14";
    case HEaaN::ParameterPreset::ST11:
        return "ST11";
    case HEaaN::ParameterPreset::ST8:
        return "ST8";
    case HEaaN::ParameterPreset::ST7:
        return "ST7";
    case HEaaN::ParameterPreset::SS7:
        return "SS7";
    case HEaaN::ParameterPreset::SD3:
        return "SD3";
    case HEaaN::ParameterPreset::CUSTOM:
        return "CUSTOM";
    case HEaaN::ParameterPreset::FVc:
        return "FVc";
    default:
        throw std::invalid_argument("Not supported parameter");
    }
}
