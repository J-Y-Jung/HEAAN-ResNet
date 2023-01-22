#include <iostream>
#include "HEaaN/heaan.hpp"
#include "oddLazyBSGS.hpp"
#include "HEaaNTimer.hpp"
#include "examples.hpp"
#include "kernelEncode.hpp"
#include "imageEncode.hpp"
#include <omp.h>
#include <time.h>

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
    enc.encrypt(msg, pack, ctxt, 4, 0);
    std::cout << "done" << std::endl;
    
    EnDecoder ecd(context);
    Ciphertext ctxt_relu(context);
    
    std::vector<double> polynomial_1 = {
    1.34595769293910e-33, 2.45589415425004e1, 4.85095667238242e-32, -6.69660449716894e2,
    -2.44541235853840e-30, 6.67299848301339e3, 1.86874811944640e-29, -3.06036656163898e4,
    -5.76227817577242e-29, 7.31884032987787e4, 8.53680673009259e-29, -9.44433217050084e4,
    -6.02701474694667e-29, 6.23254094212546e4, 1.62342843661940e-29, -1.64946744117805e4
    };

    std::vector<double> polynomial_2 = {
    1.53261588585630e-47, 9.35625636035439, -3.68972123048249e-46, -5.91638963933626e1,
    1.74254399703303e-45 , 1.48860930626448e2, -3.20672110002213e-45, -1.75812874878582e2,
    2.79115738948645e-45, 1.09111299685955e2, -1.22590309306100e-45, -3.66768839978755e1,
    2.62189142557962e-46, 6.31846290311294, -2.16662326421275e-47, -4.37113415082177e-01
    };

    std::vector<double> polynomial_3 = {
    6.43551938319983e-48, 5.07813569758861, 8.12601038855762e-46, -3.07329918137186e1,
    -1.60198474678427e-44, 1.44109746812809e2, 1.07463154460511e-43, -4.59661688826142e2,
    -3.63448723044512e-43, 1.02152064470459e3, 7.25207125369784e-43, -1.62056256708877e3,
    -9.27306397853655e-43, 1.86467646416570e3, 7.95843097354065e-43, -1.56749300877143e3,
    -4.69190103147527e-43, 9.60970309093422e2, 1.90863349654016e-43, -4.24326161871646e2,
    -5.27439678020696e-44, 1.31278509256003e2, 9.47044937974786e-45, -2.69812576626115e1,
    -9.98181561763750e-46, 3.30651387315565, 4.69390466192199e-47, -1.82742944627533e-1
    };
    
    
    //for optimization
    for(int  i = 0 ; i < 28 ; ++i){
        polynomial_3[i] = polynomial_3[i] * 0.5;
    }
    
    
    #pragma omp parallel num_threads(1)
    {
        cout << "evaluating ReLU ... \n";

        cout << "imaginary removing ...\n";
        timer.start(" * ");
        HEaaN::Ciphertext ctxt_temp(context);
        eval.conjugate(ctxt, ctxt_temp);
        eval.add(ctxt_temp, ctxt, ctxt_temp);
        eval.mult(ctxt_temp, 0.5, ctxt_temp);
        timer.end();

        cout << "1st BTS...\n";
        timer.start(" * ");
        HEaaN::Ciphertext ctxt_real_BTS(context);
        eval.bootstrap(ctxt_temp, ctxt_real_BTS, true);
        timer.end();


        //for optimization
        for(int  i = 0 ; i < 28 ; ++i){
            polynomial_3[i] = polynomial_3[i] * 0.5;
        }

        cout<< "1st poly evaluation ...\n";
        timer.start(" * ");
        evalOddPolynomial(context, eval, ctxt_real_BTS, ctxt_temp, polynomial_1, 4, 2);
        timer.end();

        cout << "2nd poly evaluation ...\n";
        timer.start(" * ");
        HEaaN::Ciphertext ctxt_temp1(context);
        evalOddPolynomial(context, eval, ctxt_temp, ctxt_temp1, polynomial_2, 2, 3);
        timer.end();

        cout << "2nd BTS ... \n";
        timer.start(" * ");
        eval.bootstrap(ctxt_temp1, ctxt_temp, true);
        timer.end();

        cout << "3rd poly evaluation ...\n";
        timer.start(" * ");
        evalOddPolynomial(context, eval, ctxt_temp, ctxt_temp1, polynomial_3, 4, 3);
        timer.end();

        cout << "evaluating for ReLU ... ";
        timer.start(" * ");
        eval.mult(ctxt_real_BTS, 0.5, ctxt_temp);
        eval.mult(ctxt_real_BTS, ctxt_temp1, ctxt_relu);
        eval.add(ctxt_temp, ctxt_relu, ctxt_relu);
        eval.levelDown(ctxt_relu, 5, ctxt_relu);
        timer.end();
    }
    
    return 0;
    
}



