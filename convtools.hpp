#pragma once
#include <iostream>
#include "HEaaN/heaan.hpp"

namespace {
    using namespace HEaaN;
    using namespace std;
}

/*weight index : (weight_row_idx, weight_col_idx)
for 3-by-3 kernel
|(0,0)  (0,1)  (0,2)|
|(1,0)  (1,1)  (1,2)|
|(2,0)  (2,1)  (2,2)|
for 1-by-1 kernel, you must set
weight index = |(1,1)|
*/

void weightToPtxt(Plaintext& ptxt, u64 level, double weight,
    u64 gap_in, u64 stride,
    u64 weight_row_idx, u64 weight_col_idx, EnDecoder ecd) {
    Message msg(15);
    auto num_slots = msg.getSize();
    size_t idx;

    if (gap_in == 1 && stride == 2) {
        //#pragma omp parallel for collapse(2)
        for (size_t k = 0; k < 32; ++k) {
            for (size_t i = 0; i < 1024; ++i) {
                msg[k * 1024 + i].imag(0.0);
                if (i % 2 == 0 && (i % 64) < 32) {
                    msg[k * 1024 + i].real(weight);
                } else {
                    msg[k * 1024 + i].real(0.0);
                }
            }
        }
    } else if (gap_in == 2 && stride == 2) {
        //#pragma omp parallel for collapse(3)
        for (size_t k = 0; k < 32; ++k) {
            for (size_t l = 0; l < 32; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    msg[k * 1024 + l * 32 + m].imag(0.0);
                    if (m % 4 < 2 && l % 4 < 2) {
                        msg[k * 1024 + l * 32 + m].real(weight);
                    } else {
                        msg[k * 1024 + l * 32 + m].real(0.0);
                    }
                }
            }
        }
    } else {
        //#pragma omp parallel for
        for (size_t j = 0; j < num_slots; ++j) {
            msg[j].real(weight);
            msg[j].imag(0.0);
        }
    }

    if (weight_row_idx == 0) {
        //#pragma omp parallel for collapse(2)
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32 * gap_in; ++i) {
                msg[j * 1024 + i].real(0.0);
            }
        }
    }
    else if (weight_row_idx == 2) {
        //#pragma omp parallel for collapse(2)
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32 * gap_in; ++i) {
                msg[j * 1024 + (1024 - 32 * gap_in) + i].real(0.0);
            }
        }
    }
    if (weight_col_idx == 0) {
        //#pragma omp parallel for collapse(3)
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                for (size_t k = 0; k < gap_in; ++k) {
                    msg[j * 1024 + i * 32 + k].real(0.0);
                }
            }
        }
    }
    else if (weight_col_idx == 2) {
        //#pragma omp parallel for collapse(3)
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                for (size_t k = 0; k < gap_in; ++k) {
                    msg[j * 1024 + i * 32 + (31 - k)].real(0.0);
                }
            }
        }
    }
    ptxt = ecd.encode(msg, level, 0);
    return;
}


