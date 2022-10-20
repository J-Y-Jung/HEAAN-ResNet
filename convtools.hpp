
#pragma once
#include <iostream>
#include "HEaaN/heaan.hpp"

namespace {
using namespace HEaaN;
using namespace std;
}
void printMessage(const Message &msg, bool is_complex = true,
                  size_t start_num = 64, size_t end_num = 64) {
    const size_t msg_size = msg.getSize();

    cout << "[ ";
    for (size_t i = 0; i < start_num; ++i) {
        if (is_complex)
            cout << msg[i] << ", ";
        else
            cout << msg[i].real() << ", ";
    }
    cout << "..., ";
    for (size_t i = end_num; i > 1; --i) {
        if (is_complex)
            cout << msg[msg_size - i] << ", ";
        else
            cout << msg[msg_size - i].real() << ", ";
    }
    if (is_complex)
        cout << msg[msg_size - 1] << " ]" << endl;
    else
        cout << msg[msg_size - 1].real() << " ]" << endl;
}


/*weight index : (weight_row_idx, weight_col_idx) 
for 3-by-3 kernel
|(0,0)  (0,1)  (0,2)|
|(1,0)  (1,1)  (1,2)|
|(2,0)  (2,1)  (2,2)|
for 1-by-1 kernel, you must set
weight index = |(1,1)|
*/

void Weight2Msg(Message &msg, vector<vector<double>> weights,
                    u64 gap_in, u64 stride, 
                    u64 weight_row_idx, u64 weight_col_idx) {
    auto num_slots = msg.getSize();
    u64 multiplicity = 2*gap_in; 
    u64 channels_in = 16*gap_in;
    //gap_in is the interval between pixels in each input image;
    //i.e. gap_in = 1 means the pixels are next to each other.
    //gap_out = (next gap_in) = stride*gap_in
    //multiplicity is a variable that indicated how many copies of input(or how many SIMD inputs) are currently in a single ctxt.

    //check if the size of the input kernel vector is valid; otherwise return error message.
    if (weights.size() != multiplicity){
        cout << "Number of input kernel values does not match!" << endl;
        return;
    }
   
    if (weights[0].size() != channels_in){
        cout << "Number of input channels does not match!" << endl;
        return;
    }
    

    //zeroize message first
    for (size_t j = 0; j < num_slots; ++j){
        msg[j].real(0.0);
        msg[j].imag(0.0);   
    }

    //define useful variables
    u64 slots_per_input = num_slots/multiplicity;
    u64 slots_per_block = 1024; //32*32 for cifar10
    size_t idx;

    //pack kernel values into message in each case
    if (gap_in==1 && stride==1){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t k = 0; k < channels_in; ++k){
                for (size_t i = 0; i < slots_per_block; ++i){
                    idx = j*slots_per_input+k*slots_per_block+i;
                    msg[idx].real(weights[j][k]);
                }
            }
        }
    }
    else if (gap_in==2 && stride==1){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t k = 0; k < channels_in; ++k){
                for (size_t l = 0; l < 16; ++l){
                    for (size_t m = 0; m < 16; ++m){
                        idx = j*slots_per_input+(k>>2)*slots_per_block+64*l+2*m+((k>>1)%2)*32+(k%2);                       
                        msg[idx].real(weights[j][k]);
                    }
                }
            }
        }
    }
    else if (gap_in==4 && stride==1){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t k = 0; k < channels_in; ++k){
                for (size_t l = 0; l < 8; ++l){
                    for (size_t m = 0; m < 8; ++m){
                        idx = j*slots_per_input+(k>>4)*slots_per_block+128*l+4*m+((k>>2)%4)*32+(k%4);                       
                        msg[idx].real(weights[j][k]);
                    }
                }
            }
        }
    }
    else if (gap_in==1 && stride==2){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t k = 0; k < channels_in; ++k){
                for (size_t i = 0; i < slots_per_block; ++i){
                    if (i%2==0 && (i%64)<32){
                        idx = j*slots_per_input+k*slots_per_block+i;                      
                        msg[idx].real(weights[j][k]);
                    }
                }
            }
        }
    }
    else if (gap_in==2 && stride==2){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t k = 0; k < channels_in; ++k){
                for (size_t l = 0; l < 16; ++l){
                    for (size_t m = 0; m < 16; ++m){
                        if (m%2==0 && l%2==0){
                            idx = j*slots_per_input+(k>>2)*slots_per_block+64*l+2*m+((k>>1)%2)*32+(k%2);                           
                            msg[idx].real(weights[j][k]);
                        }
                    }
                }
            }
        }
    }
    if (weight_row_idx == 0){
        for (size_t j = 0; j < 32; ++j){
            for (size_t i = 0; i < 32*gap_in; ++i) {
                msg[j*1024+i].real(0.0);
            }
        }
    }
    else if (weight_row_idx == 2){
        for (size_t j = 0; j < 32; ++j){
            for (size_t i = 0; i < 32*gap_in; ++i) {
                msg[j*1024+(1024-32*gap_in)+i].real(0.0);
            }
        }
    }
    if (weight_col_idx == 0){
        for (size_t j = 0; j < 32; ++j){
            for (size_t i = 0; i < 32; ++i) {
                for (size_t k = 0; k < gap_in; ++k) {
                    msg[j*1024+i*32+k].real(0.0);
                }
            }
        }
    }
    else if (weight_col_idx == 2){
        for (size_t j = 0; j < 32; ++j){
            for (size_t i = 0; i < 32; ++i) {
                for (size_t k = 0; k < gap_in; ++k) {
                    msg[j*1024+i*32+(31-k)].real(0.0);
                }
            }
        }
    }
    return;
}
