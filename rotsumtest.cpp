#include <iostream>
#include "HEaaN/heaan.hpp"
#include "HEaaNTimer.hpp"
#include "convtools.hpp"
#include "rotsum.hpp"

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

    Encryptor enc(context);
    Decryptor dec(context);

    cout << "Generate HomEvaluator ..."
              << endl;
    timer.start("* ");
    HomEvaluator eval(context, pack);
    timer.end(); 

    {
        

        Message msg(log_slots), dmsg1, dmsg2, dmsg3;
        for (size_t i = 0; i < num_slots; ++i) {
            msg[i].real((double) i);
            msg[i].imag(0.0);
        }
        printMessage(msg);
        cout << endl << "message vector: " << endl;
        cout << "Encrypt ... ";
        Ciphertext ctxt(context), ctxt_out1(context), ctxt_out2(context), ctxt_out3(context);
        enc.encrypt(msg, pack, ctxt);
        cout << "done" << endl << endl;

        cout << "Evaluating rotsum-(all sum)" << endl;
        timer.start("* ");
        ctxt_out1 = RotSumToIdx(context, pack, eval, 1, log_slots, 0, ctxt);
        timer.end(); 

        cout << "Evaluating rotsum-(block sum)" << endl;
        timer.start("* ");
        ctxt_out2 = RotSumToIdx(context, pack, eval, 1024, 4, 0, ctxt);
        timer.end(); 

        cout << "Evaluating rotsum-(block sum to idx)" << endl;
        timer.start("* ");
        ctxt_out3 = RotSumToIdx(context, pack, eval, 1024, 4, 1024*7, ctxt);
        timer.end(); 

        cout << "Decrypt ... ";
        dec.decrypt(ctxt_out1, sk, dmsg1);
        dec.decrypt(ctxt_out2, sk, dmsg2);
        dec.decrypt(ctxt_out3, sk, dmsg3);
        cout << "done" << endl;

        cout << endl << "Decrypted message(allsum): " << endl;
        cout << dmsg1[0] << endl;
        cout << endl << "Decrypted message(block sum): " << endl;
        cout << dmsg2[0] << endl;
        cout << endl << "0th idx value(trash)" << endl;
        cout << dmsg3[0] << endl;
        cout << endl << "32*32*7th idx value(block sum)" << endl;
        cout << dmsg3[1024*7] << endl;
    }

    return 0;
}