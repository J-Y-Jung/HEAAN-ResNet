
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
        Message dmsg(log_slots);
        weightToPtxt(ptxt, 5, 1.0, 1, 2, 1, 1, ecd);
        dmsg = ecd.decode(ptxt);
        cout.precision(2);
        printMessage(dmsg);
        
        weightToPtxt(ptxt, 5, 1.0, 2, 2, 1, 1, ecd);
        dmsg = ecd.decode(ptxt);
        cout.precision(2);
        printMessage(dmsg);
        
        weightToPtxt(ptxt, 5, 1.0, 1, 1, 1, 1, ecd);
        dmsg = ecd.decode(ptxt);
        cout.precision(2);
        printMessage(dmsg);
        
        weightToPtxt(ptxt, 5, 1.0, 2, 1, 1, 1, ecd);
        dmsg = ecd.decode(ptxt);
        cout.precision(2);
        printMessage(dmsg);
        
        weightToPtxt(ptxt, 5, 1.0, 4, 1, 1, 1, ecd);
        dmsg = ecd.decode(ptxt);
        cout.precision(2);
        printMessage(dmsg);

        
        
    }

    return 0;
}
