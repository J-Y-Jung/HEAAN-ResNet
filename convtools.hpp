
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
isDownsampling = true
*/
void Weight2Msg(Message &msg, vector<double> weights,
                    u64 gap_in, bool isDownsampling, 
                    u64 weight_row_idx, u64 weight_col_idx) {
    auto log_slots = msg.getLogSlots();
    auto num_slots = msg.getSize();
    u64 multiplicity = 2*gap_in; //gap_in = gap_out if !isDownsampling, otherwise gap_in*2=gap_out

    if (weights.size() != multiplicity){
        cout << "Number of input weights does not match!" << endl;
        return;
    }
    u64 slots_per_input = num_slots/multiplicity;
    if (!isDownsampling){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t i = 0; i < slots_per_input; ++i) {
                msg[j*slots_per_input+i].real(weights[j]);
                msg[j*slots_per_input+i].imag(0.0);
            }
        }
    }
    else if (gap_in==1){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t i = 0; i < slots_per_input; ++i) {
                if (i%2==0 && (i%64)<32){
                    msg[j*slots_per_input+i].real(weights[j]);
                    msg[j*slots_per_input+i].imag(0.0);
                } else {
                    msg[j*slots_per_input+i].real(0.0);
                    msg[j*slots_per_input+i].imag(0.0);
                }
            }
        }
    }
    else if (gap_in==2){
        for (size_t j = 0; j < multiplicity; ++j){
            for (size_t i = 0; i < slots_per_input; ++i) {
                if ((i%4)<2 && (i%128)<64){
                    msg[j*slots_per_input+i].real(weights[j]);
                    msg[j*slots_per_input+i].imag(0.0);
                } else {
                    msg[j*slots_per_input+i].real(0.0);
                    msg[j*slots_per_input+i].imag(0.0);
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
