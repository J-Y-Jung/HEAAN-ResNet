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

void weightToPtxtOld(Plaintext& ptxt, u64 level, vector<vector<double>> weights,
    u64 gap_in, u64 stride,
    u64 weight_row_idx, u64 weight_col_idx, EnDecoder ecd) {
    Message msg(15);
    auto num_slots = msg.getSize();
    u64 multiplicity = 2 * gap_in;
    u64 channels_in = 16 * gap_in;
    //gap_in is the interval between pixels in each input image;
    //i.e. gap_in = 1 means the pixels are next to each other.
    //gap_out = (next gap_in) = stride*gap_in
    //multiplicity is a variable that indicated how many copies of input(or how many SIMD inputs) are currently in a single ctxt.

    //check if the size of the input kernel vector is valid; otherwise return error message.
    if (weights.size() != multiplicity) {
        cout << "Number of input kernel values does not match!" << endl;
        return;
    }

    if (weights[0].size() != channels_in) {
        cout << "Number of input channels does not match!" << endl;
        return;
    }


    //zeroize message first
    for (size_t j = 0; j < num_slots; ++j) {
        msg[j].real(0.0);
        msg[j].imag(0.0);
    }

    //define useful variables
    u64 slots_per_input = num_slots / multiplicity;
    u64 slots_per_block = 1024; //32*32 for cifar10
    size_t idx;

    //pack kernel values into message in each case
    if (gap_in == 1 && stride == 1) {
        for (size_t j = 0; j < multiplicity; ++j) {
            for (size_t k = 0; k < channels_in; ++k) {
                for (size_t i = 0; i < slots_per_block; ++i) {
                    idx = j * slots_per_input + k * slots_per_block + i;
                    msg[idx].real(weights[j][k]);
                }
            }
        }
    }
    else if (gap_in == 2 && stride == 1) {
        for (size_t j = 0; j < multiplicity; ++j) {
            for (size_t k = 0; k < channels_in; ++k) {
                for (size_t l = 0; l < 16; ++l) {
                    for (size_t m = 0; m < 16; ++m) {
                        idx = j * slots_per_input + (k >> 2) * slots_per_block + 64 * l + 2 * m + ((k >> 1) % 2) * 32 + (k % 2);
                        msg[idx].real(weights[j][k]);
                    }
                }
            }
        }
    }
    else if (gap_in == 4 && stride == 1) {
        for (size_t j = 0; j < multiplicity; ++j) {
            for (size_t k = 0; k < channels_in; ++k) {
                for (size_t l = 0; l < 8; ++l) {
                    for (size_t m = 0; m < 8; ++m) {
                        idx = j * slots_per_input + (k >> 4) * slots_per_block + 128 * l + 4 * m + ((k >> 2) % 4) * 32 + (k % 4);
                        msg[idx].real(weights[j][k]);
                    }
                }
            }
        }
    }
    else if (gap_in == 1 && stride == 2) {
        for (size_t j = 0; j < multiplicity; ++j) {
            for (size_t k = 0; k < channels_in; ++k) {
                for (size_t i = 0; i < slots_per_block; ++i) {
                    if (i % 2 == 0 && (i % 64) < 32) {
                        idx = j * slots_per_input + k * slots_per_block + i;
                        msg[idx].real(weights[j][k]);
                    }
                }
            }
        }
    }
    else if (gap_in == 2 && stride == 2) {
        for (size_t j = 0; j < multiplicity; ++j) {
            for (size_t k = 0; k < channels_in; ++k) {
                for (size_t l = 0; l < 16; ++l) {
                    for (size_t m = 0; m < 16; ++m) {
                        if (m % 2 == 0 && l % 2 == 0) {
                            idx = j * slots_per_input + (k >> 2) * slots_per_block + 64 * l + 2 * m + ((k >> 1) % 2) * 32 + (k % 2);
                            msg[idx].real(weights[j][k]);
                        }
                    }
                }
            }
        }
    }
    if (weight_row_idx == 0) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32 * gap_in; ++i) {
                msg[j * 1024 + i].real(0.0);
            }
        }
    }
    else if (weight_row_idx == 2) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32 * gap_in; ++i) {
                msg[j * 1024 + (1024 - 32 * gap_in) + i].real(0.0);
            }
        }
    }
    if (weight_col_idx == 0) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                for (size_t k = 0; k < gap_in; ++k) {
                    msg[j * 1024 + i * 32 + k].real(0.0);
                }
            }
        }
    }
    else if (weight_col_idx == 2) {
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

void weightToPtxt(Plaintext& ptxt, u64 level, double weight,
    u64 gap_in, u64 stride,
    u64 weight_row_idx, u64 weight_col_idx, EnDecoder ecd) {
    Message msg(15);
    auto num_slots = msg.getSize();
    size_t idx;

    if (gap_in == 1 && stride == 2) {
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < 32; ++k) {
            for (size_t i = 0; i < 1024; ++i) {
                idx = k * 1024 + i;
                msg[idx].imag(0.0);
                if (i % 2 == 0 && (i % 64) < 32) {
                    msg[idx].real(weight);
                }
                else {
                    msg[idx].real(0.0);
                }
            }
        }
    }
    else if (gap_in == 2 && stride == 2) {
        #pragma omp parallel for collapse(3)
        for (size_t k = 0; k < 32; ++k) {
            for (size_t l = 0; l < 32; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    idx = k * 1024 + l * 32 + m;
                    msg[idx].imag(0.0);
                    if (m % 4 < 2 && l % 4 < 2) {
                        msg[idx].real(weight);
                    }
                    else {
                        msg[idx].real(0.0);
                    }
                }
            }
        }
    }
    else {
        #pragma omp parallel for
        for (size_t j = 0; j < num_slots; ++j) {
            msg[j].real(weight);
        }
    }

    if (weight_row_idx == 0) {
        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32 * gap_in; ++i) {
                msg[j * 1024 + i].real(0.0);
            }
        }
    }
    else if (weight_row_idx == 2) {
        #pragma omp parallel for collapse(2)
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32 * gap_in; ++i) {
                msg[j * 1024 + (1024 - 32 * gap_in) + i].real(0.0);
            }
        }
    }
    if (weight_col_idx == 0) {
        #pragma omp parallel for collapse(3)
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                for (size_t k = 0; k < gap_in; ++k) {
                    msg[j * 1024 + i * 32 + k].real(0.0);
                }
            }
        }
    }
    else if (weight_col_idx == 2) {
        #pragma omp parallel for collapse(3)
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

//weight for the first conv layer
void weightToPtxt1stConvOld(Plaintext& ptxt, u64 level, vector<double> weights,
    u64 weight_row_idx, u64 weight_col_idx, EnDecoder ecd) {
    Message msg(15);
    auto num_slots = msg.getSize();
    u64 multiplicity = 32;
    u64 channels_in = 1;

    if (weights.size() != channels_in) {
        cout << "Number of input channels does not match!" << endl;
        return;
    }

    //zeroize message first
    for (size_t j = 0; j < num_slots; ++j) {
        msg[j].real(0.0);
        msg[j].imag(0.0);
    }

    //define useful variables
    u64 slots_per_input = num_slots / multiplicity;
    u64 slots_per_block = 1024; //32*32 for cifar10
    size_t idx;

    //pack kernel values into message in each case

    for (size_t j = 0; j < multiplicity; ++j) {
        for (size_t k = 0; k < channels_in; ++k) {
            for (size_t i = 0; i < slots_per_block; ++i) {
                idx = j * slots_per_input + k * slots_per_block + i;
                msg[idx].real(weights[j]);
            }
        }
    }

    if (weight_row_idx == 0) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                msg[j * 1024 + i].real(0.0);
            }
        }
    }
    else if (weight_row_idx == 2) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                msg[j * 1024 + (1024 - 32) + i].real(0.0);
            }
        }
    }
    if (weight_col_idx == 0) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                msg[j * 1024 + i * 32].real(0.0);
            }
        }
    }
    else if (weight_col_idx == 2) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t i = 0; i < 32; ++i) {
                msg[j * 1024 + i * 32 + 31].real(0.0);
            }
        }
    }
    ptxt = ecd.encode(msg, level, 0);
    return;
}