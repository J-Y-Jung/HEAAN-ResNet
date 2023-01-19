#include <iostream>
#include "HEaaN/heaan.hpp"
#include "oddLazyBSGS.hpp"
#include "HEaaNTimer.hpp"
#include "examples.hpp"
#include "kernelEncode.hpp"
#include "imageEncode.hpp"
#include <omp.h>

namespace {
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
        msg[i].real((double)(pow(-1, i)));
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


    cout << "test for evalPoly..." <<"\n";
    constexpr const HEaaN::Real POLY_COEFF[] = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ,1 };

    vector<double> polynomial = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ,1 };

    constexpr const size_t POLY_DEGREE = 15;


    std::cout << "Input ciphertext - level " << ctxt.getLevel()
        << std::endl
        << std::endl;

    std::cout << "Evaluating odd polynomial of the form" << std::endl;
    print_polynomial(POLY_COEFF, POLY_DEGREE);
    std::cout << std::endl << " ... ";

    std::cout << "Start evaluating odd polynomial ... " << std::endl;


    Ciphertext ctxt_out1(context);
    timer.start("* ");
    evalOddPolynomial(context, eval, ctxt, ctxt_out1, polynomial, 4, 2);
    timer.end();
    std::cout << "done" << std::endl << std::endl;

    std::cout << "Output ciphertext of polynomial - level " << ctxt_out1.getLevel()
        << std::endl
        << std::endl;

    HEaaN::Message dmsg1;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_out1, sk, dmsg1);
    std::cout << "done" << std::endl;

    std::cout << std::endl << "Decrypted result vector : " << std::endl;
    printMessage(dmsg1, false);

    
    Plaintext ptxt_init(context);
    double cnst =(double)(1.0/40.0);

    vector<double> temp7;
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path7 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block4conv_onebyone_multiplicands32_16_1_1");
    txtreader(temp7, path7);
    kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 5, 1, 2, 32, 16, 1, ecd);
    temp7.clear();
    temp7.shrink_to_fit();
    
    printMessage(ecd.decode(block4conv_onebyone_multiplicands32_16_1_1[0][0][0]));
    printMessage(ecd.decode(block4conv_onebyone_multiplicands32_16_1_1[0][0][1]));
    
    vector<Plaintext> block4conv_onebyone_summands32;
    vector<double> temp7a;
    string path7a = "/app/HEAAN-ResNet/kernel/summands/" + string("block4conv_onebyone_summands32");
    Scaletxtreader(temp7a, path7a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp7a[i]);
        block4conv_onebyone_summands32.push_back(ecd.encode(msg, 4, 0));
    }
    temp7a.clear();
    temp7a.shrink_to_fit();
    
    printMessage(ecd.decode(block4conv_onebyone_summands32[0]));
    printMessage(ecd.decode(block4conv_onebyone_summands32[1]));
    
    return 0;
    
    
}



