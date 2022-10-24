#include <iostream>
#include "HEaaN/heaan.hpp"
#include "evalpoly.hpp"
#include "HEaaNTimer.hpp"
#include "examples.hpp"

namespace {
    using namespace std;
    using namespace HEaaN;
}

//test evalpoly part.
int main(){
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

    cout << "Generate commonly used keys (mult key, rotation keys, "
                 "conjugation key) ... "
              << endl;
    keygen.genCommonKeys();
    cout << "done" << endl << endl;

    Encryptor enc(context);
    Decryptor dec(context);

    cout << "Generate HomEvaluator (including pre-computing constants for "
                 "bootstrapping) ..."
              << endl;
    
    HomEvaluator eval(context, pack);
    cout << "done" << endl << endl;

    //test for AReLu.
    Ciphertext ctxt(context);

    cout << "Evaluating approximate ReLU function homomorphically ..."
              << endl;
    
    timer.start("* ");
    ApproxReLU(context, eval, ctxt);
    timer.end();

    return 0;

}

