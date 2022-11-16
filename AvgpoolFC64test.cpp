#include <iostream>
#include "HEaaN/heaan.hpp"
#include "HEaaNTimer.hpp"
#include "AvgpoolFC64.hpp"
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

    cout << "Generate commonly used keys (mult key, rotation keys, "
                 "conjugation key) ... "
              << endl;
    keygen.genCommonKeys();
    cout << "done" << endl << endl;

    EnDecoder ecd(context);
    Encryptor enc(context);
    Decryptor dec(context);

    cout << "Generate HomEvaluator ..."
              << endl;
    timer.start("* ");
    HomEvaluator eval(context, pack);
    timer.end(); 

    {
        Message msg(log_slots), uni(log_slots, COMPLEX_ZERO), dmsg(log_slots), msg_tmp(log_slots), 
                msg_tmp0(log_slots, COMPLEX_ZERO), msg_out(log_slots, COMPLEX_ZERO);
        Plaintext ptxt(context);
        vector<Plaintext> ptxt_vec;
        for (size_t i = 0; i < num_slots; ++i) {
            msg[i] = 1;
        }
        cout << "original message: " << endl;
        printMessage(msg, false, 64, 64);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    uni[i+32*j+32*32*k] = 1.0;
                }
            }
        }
        cout << "mask: " << endl;
        printMessage(uni, false, 64, 64);
        Ciphertext ctxt(context), ctxt_out(context);
        enc.encrypt(msg, pack, ctxt);
        ctxt.setLevel(5);
        cout << "evaluating Avgpool" << endl;
        timer.start("* ");
        ctxt = Avgpool(context, pack, eval, ctxt);
        timer.end();
        dec.decrypt(ctxt, sk, dmsg);
        cout << "avgpool message: " << endl;
        printMessage(dmsg, false, 64, 64);

        ptxt = ecd.encode(uni, 5, 0);
        for (size_t i = 0; i < 10; ++i) {
            ptxt_vec.push_back(ptxt);
        }
        cout << "evaluating FC64" << endl;
        timer.start("* ");
        ctxt_out = FC64(context, pack, eval, ctxt, ptxt_vec);
        timer.end();
        dec.decrypt(ctxt_out, sk, dmsg);
        cout << "decrypted message after FC64: " << endl;
        printMessage(dmsg, false, 64, 64);
        //(0, 8, 16, 24, 256, 264, 272, 280, 512, 520)
        cout << "actual result:" << endl << "[ ";
        cout << dmsg[0].real() << ", "<< dmsg[8].real() << ", "<< dmsg[16].real() << ", "<< dmsg[24].real() << ", "
        << dmsg[256].real() << ", "<< dmsg[264].real() << ", "<< dmsg[272].real() << ", "<< dmsg[280].real() << ", "
        << dmsg[512].real() << ", "<< dmsg[520].real() << " ]" << endl;
        //must be all same

        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                eval.leftRotate(msg, 4*i+32*4*j, msg_tmp);
                eval.add(msg_tmp0, msg_tmp, msg_tmp0);
            }
        }
        eval.mult(msg_tmp0, uni, msg_tmp0);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    eval.leftRotate(msg_tmp0, i+32*j+32*32*k, msg_tmp);
                    eval.add(msg_out, msg_tmp, msg_out);
                }
            }
        }
        cout << "target value: " << endl;
        cout << msg_out[0] << endl;
        //must be equal to the actual result value

    }
    return 0;
}