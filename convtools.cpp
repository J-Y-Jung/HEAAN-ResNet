/*
Todo : 
make this as header file (hpp) 
integrate with I/O
make a function that takes weights, BN values, scaling constants and outputs modified CNN weights
*/

#include <iostream>
#include "HEaaN/heaan.hpp"
#include "HEaaNTimer.hpp"

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


//test code
int main() {
    HEaaNTimer timer(false);
    ParameterPreset preset = ParameterPreset::FGb;
    Context context = makeContext(preset);

    const auto log_slots = getLogFullSlots(context);
    const auto num_slots = U64C(1) << log_slots;

    cout << "Generate secret key ... ";
        SecretKey sk(context);
        cout << "done" << endl;

    KeyPack pack(context);
    KeyGenerator keygen(context, sk, pack);

    cout << "Generate encryption key ... " << endl;
    keygen.genEncryptionKey();
    cout << "done" << endl << endl;

    makeBootstrappable(context);

    /*cout << "Generate commonly used keys (mult key, rotation keys, "
                 "conjugation key) ... "
              << endl;
    keygen.genCommonKeys();
    cout << "done" << endl << endl;*/

    Encryptor enc(context);
    Decryptor dec(context);

    cout << "Generate HomEvaluator (including pre-computing constants for "
                 "bootstrapping) ..."
              << endl;
    timer.start("* ");
    HomEvaluator eval(context, pack);
    timer.end();

    {
        Message wgt(log_slots);
        vector<double> v(2);
        v[0]=1.2; v[1]=3.4;
        u64 gap_in = 1;
        bool isDownsampling = true;
        u64 weight_row_idx = 1; 
        u64 weight_col_idx = 1;
        Weight2Msg(wgt, v, gap_in, isDownsampling, weight_row_idx, weight_col_idx);
        cout << endl << "output weight vector: " << endl;
        printMessage(wgt);
        cout << endl;

        Message msg(log_slots);
        for (size_t i = 0; i < num_slots; ++i) {
            msg[i].real((double)((197*i*i) %256)/256 - 0.5);
            msg[i].imag(0.0);
        }
        printMessage(msg);
        cout << endl << "message vector: " << endl;
        cout << "Encrypt ... ";
        Ciphertext ctxt(context), ctxt_out(context);
        enc.encrypt(msg, pack, ctxt);
        cout << "done" << endl << endl;

        cout << "Evaluating weight * ctxt" << endl;
        eval.mult(ctxt, wgt, ctxt_out);
        cout << "done" << endl << endl;

        cout << "Result ciphertext - level " << ctxt_out.getLevel()
                  << endl
                  << endl;

        Message dmsg;
        cout << "Decrypt ... ";
        dec.decrypt(ctxt_out, sk, dmsg);
        cout << "done" << endl;

        cout.precision(2);
        cout << endl << "Decrypted message : " << endl;
        printMessage(dmsg);
    }

    return 0;
}