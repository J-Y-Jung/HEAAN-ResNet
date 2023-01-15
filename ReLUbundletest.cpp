#include <iostream>
#include "HEaaN/heaan.hpp"
//#include "oddLazyBSGS.hpp"
#include "HEaaNTimer.hpp"
#include "examples.hpp"
#include "ReLUbundle.hpp"

namespace{
    using namespace std;
    using namespace HEaaN;
}

void print_polynomial(const double* coeff, const size_t degree) {
    auto print_coeff = [](double coeff) {
        if (coeff < 0.0L)
            std::cout << " - " << std::abs(coeff);
        else
            std::cout << " + " << coeff;
    };

    std::cout << coeff[degree] << "X^" << degree;
    for (size_t idx = degree - 1; idx > 1; --idx) {
        print_coeff(coeff[idx]);
        std::cout << "X^" << idx;
    }
    print_coeff(coeff[1]);
    std::cout << "X";
    print_coeff(coeff[0]);
}

//test evalpoly part.
int main() {
    HEaaN::HEaaNTimer timer(false);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    if (!HEaaN::isBootstrappableParameter(context)) {
        std::cout << "Bootstrap is not available for parameter "
            << presetNamer(preset) << std::endl;
        return -1;
    }

    std::cout << "Parameter : " << presetNamer(preset) << std::endl
        << std::endl;

    const auto log_slots = getLogFullSlots(context);

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... " << std::endl;
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl << std::endl;

    /*
    You should perform makeBootstrappble function
    before generating evaluation keys and constucting HomEvaluator class.
    */
    HEaaN::makeBootstrappable(context);

    std::cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << std::endl;
    keygen.genCommonKeys();
    std::cout << "done" << std::endl << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);

    /*
    HomEvaluator constructor pre-compute the constants for bootstrapping.
    */
    std::cout << "Generate HomEvaluator (including pre-computing constants for "
        "bootstrapping) ..."
        << std::endl;
    timer.start("* ");
    HEaaN::HomEvaluator eval(context, pack);
    timer.end();

    HEaaN::Message msg(log_slots);

    const auto num_slots = pow(2, log_slots);

    std::cout << std::endl << "Number of slots = " << num_slots << std::endl;

    for (size_t i = 0; i < num_slots; ++i) {
        msg[i].real((double)pow(-1, i));
        msg[i].imag(0.0);
    }

    std::cout << std::fixed;
    std::cout.precision(7);
    std::cout << std::endl << "Input vector : " << std::endl;
    printMessage(msg, false); // print the numbers with only real part
    std::cout << std::endl;

    HEaaN::Ciphertext ctxt(context);

    std::cout << "Shell ciphertext - level " << ctxt.getLevel()
        << std::endl
        << std::endl;

    std::cout << "Encrypt ... ";
    enc.encrypt(msg, pack, ctxt);
    std::cout << "done" << std::endl;

    int n = 10;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_bundle(n, std::vector<HEaaN::Ciphertext>(n,ctxt));

    Ciphertext ctxt_out2(context);
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out_bundle(n, std::vector<HEaaN::Ciphertext>(n,ctxt_out2));


    
    timer.start("normal");
    for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
            ApproxReLU(context,eval,ctxt_bundle[i][j] , ctxt_out_bundle[i][j]);
        }
    }
    timer.end();
    

    /*
    timer.start("parallel normal");
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
            ApproxReLU(context,eval,ctxt_bundle[i][j] , ctxt_out_bundle[i][j]);
        }
    }
    timer.end();
    */




   // Ciphertext ctxt_out2(context);
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out_bundle2(n,std::vector<HEaaN::Ciphertext>(n,ctxt_out2));

    timer.start("bundle ");
    ApproxReLU_bundle(context, eval, ctxt_bundle, ctxt_out_bundle2);
    timer.end();

    std::cout << "Output ciphertext of approximate ReLU - level " << ctxt_out_bundle[0][0].getLevel()
        << std::endl
        << std::endl;

    HEaaN::Message dmsg2;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_out_bundle[0][0], sk, dmsg2);
    std::cout << "done" << std::endl;

    std::cout << std::endl << "Decrypted result vector : " << std::endl;
    printMessage(dmsg2, false);

    HEaaN::Message dmsg3;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_out_bundle2[0][0], sk, dmsg3);
    std::cout << "done" << std::endl;

    std::cout << std::endl << "Decrypted result vector : " << std::endl;
    printMessage(dmsg3, false);

    return 0;

}

