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

    //makeBootstrappable(context);

    /*cout << "Generate commonly used keys (mult key, rotation keys, "
                 "conjugation key) ... "
              << endl;
    keygen.genCommonKeys();
    cout << "done" << endl << endl;*/
    EnDecoder ecd(context);
    Encryptor enc(context);
    Decryptor dec(context);

    cout << "Generate HomEvaluator ..."
              << endl;
    timer.start("* ");
    HomEvaluator eval(context, pack);
    timer.end(); 

    {
        Plaintext ptxt(context);
        Message wgt(log_slots);
        vector<double> v0(32);
        for (size_t i = 0; i < 32; ++i) {
            v0[i] = i+1;
        }
        vector<vector<double>> v(4, v0);
        vector<double> v1(16);
        for (size_t i = 0; i < 16; ++i) {
            v1[i] = i+1;
        }
        vector<vector<double>> w(2, v1);
        vector<double> v2(64);
        for (size_t i = 0; i < 64; ++i) {
            v2[i] = i+1;
        }
        vector<vector<double>> y(8, v2);
        weightToPtxt(ptxt,13, w, 1,1,0,0, ecd);
        cout << endl << "output weight vector: " << endl;
        wgt = ecd.decode(ptxt);
        printMessage(wgt);
        cout << endl;
        weightToPtxt(ptxt,13, w, 1,2,0,0, ecd);
        cout << endl << "output weight vector: " << endl;
        wgt = ecd.decode(ptxt);
        printMessage(wgt);
        cout << endl;
        weightToPtxt(ptxt,13, v, 2,1,1,1, ecd);
        cout << endl << "output weight vector: " << endl;
        wgt = ecd.decode(ptxt);
        printMessage(wgt);
        cout << endl;
        weightToPtxt(ptxt,13, y, 4,1,1,1, ecd);
        cout << endl << "output weight vector: " << endl;
        wgt = ecd.decode(ptxt);
        printMessage(wgt);
        cout << endl;
        weightToPtxt(ptxt,13, v, 2,2,1,1, ecd);
        cout << endl << "output weight vector: " << endl;
        wgt = ecd.decode(ptxt);
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
        eval.mult(ctxt, ptxt, ctxt_out);
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