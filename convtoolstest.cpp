/*
Todo : 
integrate with I/O
make a function that takes weights, BN values, scaling constants and outputs modified CNN weights
*/

#include <iostream>
#include "HEaaN/heaan.hpp"
#include "HEaaNTimer.hpp"
#include "convtools.hpp"

namespace {
using namespace HEaaN;
using namespace std;
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