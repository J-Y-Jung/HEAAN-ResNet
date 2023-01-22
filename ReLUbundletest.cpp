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
    enc.encrypt(msg, pack, ctxt, 5, 0);
    std::cout << "done" << std::endl;
    
    
    Ciphertext ctxt_init(context);
    Message dmsg;

//     timer.start("normal");
//     for(int i = 0 ; i < 4 ; i++){
//         for(int j = 0 ; j < 32 ; j++){
//             ApproxReLU(context,eval,ctxt_bundle[i][j] , ctxt_out_bundle[i][j]);
//         }
//     }
//     timer.end();
    
//     dec.decrypt(ctxt_out_bundle[0][0], sk, dmsg);
//     printMessage(dmsg);
    
    
    cout << "Test for evaluating ReLU for (16, 16) ctxt bundle...\n\n";
    
    vector<vector<Ciphertext>> ctxt_bundle1(16, vector<Ciphertext>(16, ctxt));
    vector<vector<Ciphertext>> ctxt_out_bundle1(16, vector<Ciphertext>(16, ctxt_init));
    
    timer.start("method 1");
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < 16 ; i++){
        for(int j = 0 ; j < 16 ; j++){
            ApproxReLU(context, eval, ctxt_bundle1[i][j] , ctxt_out_bundle1[i][j]);
        }
    }
    timer.end();
    
    timer.start("method 2");
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 80; ++i) {
        ApproxReLU(context, eval, ctxt_bundle1[i / 16][i % 16], ctxt_out_bundle1[i / 16][i % 16]);
    }
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 80; ++i) {
        ApproxReLU(context, eval, ctxt_bundle1[5 + (i / 16)][i % 16], ctxt_out_bundle1[5 + (i / 16)][i % 16]);
    }
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 80; ++i) {
        ApproxReLU(context, eval, ctxt_bundle1[10 + (i / 16)][i % 16], ctxt_out_bundle1[10 + (i / 16)][i % 16]);
    }
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
            ApproxReLU(context, eval, ctxt_bundle1[15][i % 16], ctxt_out_bundle1[15][i % 16]);
        }
    }
    timer.end();
    
    timer.start("method 3");
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 80; ++i) {
        ApproxReLU(context, eval, ctxt_bundle1[i / 16][i % 16], ctxt_out_bundle1[i / 16][i % 16]);
    }
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 80; ++i) {
        ApproxReLU(context, eval, ctxt_bundle1[5 + (i / 16)][i % 16], ctxt_out_bundle1[5 + (i / 16)][i % 16]);
    }
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 80; ++i) {
        ApproxReLU(context, eval, ctxt_bundle1[10 + (i / 16)][i % 16], ctxt_out_bundle1[10 + (i / 16)][i % 16]);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < 16; ++i) {
        ApproxReLU(context, eval, ctxt_bundle1[15][i % 16], ctxt_block3relu1_out[15][i % 16]);
    }
    timer.end();
    
    ctxt_bundle1.clear();
    ctxt_bundle1.shrink_to_fit();
    ctxt_out_bundle1.clear();
    ctxt_out_bundle1.shrink_to_fit();
    
    
    cout << "Test for evaluating ReLU for (4, 32) ctxt bundle...\n\n";
    
    vector<vector<Ciphertext>> ctxt_bundle2(4, vector<Ciphertext>(32, ctxt));
    vector<vector<Ciphertext>> ctxt_out_bundle2(4, vector<Ciphertext>(32, ctxt_init));
    
    timer.start("method 1");
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < 4 ; i++){
        for(int j = 0 ; j < 32 ; j++){
            ApproxReLU(context, eval, ctxt_bundle2[i][j] , ctxt_out_bundle2[i][j]);
        }
    }
    timer.end();
    
    timer.start("method 2");
    #pragma omp parallel for num_threads(80)
    for(int i = 0 ; i < 80 ; ++i){
        ApproxReLU(context, eval, ctxt_bundle2[i/20][(i%20)] , ctxt_out_bundle2[i/20][(i%20)]);
    }
    
    #pragma omp parallel for num_threads(48)
    for(int i = 0 ; i < 48 ; ++i){
        ApproxReLU(context, eval, ctxt_bundle2[i/12][20+(i%12)] , ctxt_out_bundle2[i/12][20+(i%12)]);
    }
    timer.end();
    
    timer.start("method 3");
    #pragma omp parallel for num_threads(80)
    for(int i = 0 ; i < 80 ; ++i){
        ApproxReLU(context, eval, ctxt_bundle2[i/20][(i%20)] , ctxt_out_bundle2[i/20][(i%20)]);
    }
    
    #pragma omp parallel for
    for(int i = 0 ; i < 48 ; ++i){
        ApproxReLU(context, eval, ctxt_bundle2[i/12][20+(i%12)] , ctxt_out_bundle2[i/12][20+(i%12)]);
    }
    timer.end();
    
    
    ctxt_bundle2.clear();
    ctxt_bundle2.shrink_to_fit();
    ctxt_out_bundle2.clear();
    ctxt_out_bundle2.shrink_to_fit();

    
    cout << "Test for evaluating ReLU for (1, 64) ctxt bundle...\n\n";
    
    vector<vector<Ciphertext>> ctxt_bundle3(1, vector<Ciphertext>(64, ctxt));
    vector<vector<Ciphertext>> ctxt_out_bundle3(1, vector<Ciphertext>(64, ctxt_init));
    
    timer.start("method 1");
    #pragma omp parallel for
    for(int j = 0 ; j < 64 ; j++){
        ApproxReLU(context, eval, ctxt_bundle3[0][j] , ctxt_out_bundle3[0][j]);
    }
    timer.end();
    
    timer.start("method 2");
    #pragma omp parallel for num_threads(80)
    for(int j = 0 ; j < 64 ; j++){
        ApproxReLU(context, eval, ctxt_bundle3[0][j] , ctxt_out_bundle3[0][j]);
    }
    timer.end();
    
    timer.start("method 3");
    #pragma omp parallel for num_threads(64)
    for(int j = 0 ; j < 64 ; j++){
        ApproxReLU(context, eval, ctxt_bundle3[0][j] , ctxt_out_bundle3[0][j]);
    }
    timer.end();
    
    ctxt_bundle3.clear();
    ctxt_bundle3.shrink_to_fit();
    ctxt_out_bundle3.clear();
    ctxt_out_bundle3.shrink_to_fit();
    

    return 0;

}

