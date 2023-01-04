#include "examples.hpp"
#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "DSB.hpp"

namespace {
using namespace HEaaN;
using namespace std;
}

int main() {
    // SetUp
    HEaaNTimer timer(false);
    ParameterPreset preset = ParameterPreset::FGb;
    Context context = makeContext(preset);
    if (!isBootstrappableParameter(context)) {
        cout << "Bootstrap is not available for parameter "
            << presetNamer(preset) << endl;
        return -1;
    }
    cout << "Parameter : " << presetNamer(preset) << endl
        << endl;
    const auto log_slots = getLogFullSlots(context);

    SecretKey sk(context);
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
    timer.start("* ");
    HomEvaluator eval(context, pack);
    timer.end();




    ///////////// Message ///////////////////
    Message msg(log_slots);
    // fillRandomComplex(msg);
    optional<size_t> num;
    size_t length = num.has_value() ? num.value() : msg.getSize();
    size_t idx = 0;
    for (; idx < length; ++idx) {
        msg[idx].real(0.5);
        msg[idx].imag(0.0);
    }
    // If num is less than the size of msg,
    // all remaining slots are zero.
    for (; idx < msg.getSize(); ++idx) {
        msg[idx].real(0.0);
        msg[idx].imag(0.0);
    }
    // printMessage(msg);

    Ciphertext ctxt(context);
    cout << "Encrypt ... " << endl;
    enc.encrypt(msg, pack, ctxt); // public key encryption
    cout << "done" << endl;


    /////////////// Kernel ////////////////
    // timer.start(" Kernel generation ");
    cout << "Kernel generation ... " << endl;
    timer.start("* ");
    EnDecoder ecd(context);
    Plaintext ptxt(context);
    ptxt = ecd.encode(msg, 12, 0);

    vector<vector<vector<Plaintext>>> kernel_o;
    // #pragma omp parallel for
    for (int o = 0; o < 32; ++o) { // output channel 32
        vector<vector<Plaintext>> kernel_i;
        // #pragma omp parallel for
        for (int i = 0; i < 16; ++i) {   // input channel 16
            vector<Plaintext> kernel_bundle;
            Message kernel_msg(log_slots);
            Plaintext kernel(context);
            // #pragma omp parallel for
            for (int k = 0; k < 9; ++k) {
                idx = 0;
                for (; idx < length; ++idx) {
                    kernel_msg[idx].real(2.0);
                    kernel_msg[idx].imag(0.0);
                }
                // If num is less than the size of msg,
                // all remaining slots are zero.
                for (; idx < kernel_msg.getSize(); ++idx) {
                    kernel_msg[idx].real(0.0);
                    kernel_msg[idx].imag(0.0);
                }
                kernel = ecd.encode(kernel_msg, 12, 0);
                kernel_bundle.push_back(kernel);
            }
            kernel_i.push_back(kernel_bundle);
        }
        kernel_o.push_back(kernel_i);
    }

    vector<vector<vector<Plaintext>>> kernel_o2;
    // #pragma omp parallel for
    for (int o = 0; o < 32; ++o) { // output channel 32
        vector<vector<Plaintext>> kernel_i2;
        for (int i = 0; i < 32; ++i) {   // input channel 32
            vector<Plaintext> kernel_bundle2;
            Message kernel_msg2(log_slots);
            Plaintext kernel2(context);
            for (int k = 0; k < 9; ++k) {
                idx = 0;
                for (; idx < length; ++idx) {
                    kernel_msg2[idx].real(2.0);
                    kernel_msg2[idx].imag(0.0);
                }
                // If num is less than the size of msg,
                // all remaining slots are zero.
                for (; idx < kernel_msg2.getSize(); ++idx) {
                    kernel_msg2[idx].real(0.0);
                    kernel_msg2[idx].imag(0.0);
                }
                kernel2 = ecd.encode(kernel_msg2, 6, 0);
                kernel_bundle2.push_back(kernel2);
            }
            kernel_i2.push_back(kernel_bundle2);
        }
        kernel_o2.push_back(kernel_i2);
    }

    vector<vector<vector<Plaintext>>> kernel_o3;
    // #pragma omp parallel for
    for (int o = 0; o < 32; ++o) { // output channel 32
        vector<vector<Plaintext>> kernel_i3;
        for (int i = 0; i < 16; ++i) {   // input channel 16
            vector<Plaintext> kernel_bundle3;
            Message kernel_msg3(log_slots);
            Plaintext kernel3(context);
            for (int k = 0; k < 1; ++k) {
                idx = 0;
                for (; idx < length; ++idx) {
                    kernel_msg3[idx].real(2.0);
                    kernel_msg3[idx].imag(0.0);
                }
                // If num is less than the size of msg,
                // all remaining slots are zero.
                for (; idx < kernel_msg3.getSize(); ++idx) {
                    kernel_msg3[idx].real(0.0);
                    kernel_msg3[idx].imag(0.0);
                }
                kernel3 = ecd.encode(kernel_msg3, 12, 0);
                kernel_bundle3.push_back(kernel3);
            }
            kernel_i3.push_back(kernel_bundle3);
        }
        kernel_o3.push_back(kernel_i3);
    }
    cout << "done" << endl;
    timer.end();



    vector<vector<Ciphertext>> ctxt_bundle;
    // #pragma omp parallel for
    for (int i = 0; i < 16; ++i) {
        vector<Ciphertext> ctxt_bundle_cache;
        for (int ch = 0; ch < 3; ++ch) {
            ctxt_bundle_cache.push_back(ctxt);
        }
        ctxt_bundle.push_back(ctxt_bundle_cache);
    }

    cout << "SETUP is over" << "\n";

    // Convolution 1
    cout << "Convolution 1 ..." << endl;
    timer.start(" Convolution 1 ");
    vector<vector<Ciphertext>> ctxt_conv1_out_bundle;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        vector<Ciphertext> ctxt_conv1_out_cache;
        ctxt_conv1_out_cache = Conv(context, pack, eval, 32, 1, 1, 3, 16, ctxt_bundle[i], kernel_bundle);
        ctxt_conv1_out_bundle.push_back(ctxt_conv1_out_cache);
    }
    timer.end();
    cout << "DONE!" << "\n";


    // AppReLU
    cout << "AppReLU ..." << endl;
    timer.start(" AppReLU 1 ");
    vector<vector<Ciphertext>> ctxt_relu1_out_bundle;
    for (int i = 0; i < 4; ++i) {
        vector<Ciphertext> ctxt_relu1_out_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            cout << "(i = " << i << ", " << "ch = " << ch << ")" << "\n";
            Ciphertext ctxt_relu1_out(context);
            ApproxReLU(context, eval, ctxt_conv1_out_bundle[i][ch], ctxt_relu1_out);
            ctxt_relu1_out_allch_bundle.push_back(ctxt_relu1_out);
        }
        ctxt_relu1_out_bundle.push_back(ctxt_relu1_out_allch_bundle);
    }
    timer.end();
    cout << "DONE!" << "\n";


    // Residual Block 1, 2, 3
    cout << "RB 1 ..." << endl;
    timer.start(" RB 1 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB1_out = RB(context, pack, eval, 0, ctxt_relu1_out_bundle, kernel_o, kernel_o2);
    timer.end();
    cout << "DONE!" << "\n";

    cout << "RB 2..." << endl;
    timer.start(" RB 2 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB2_out = RB(context, pack, eval, 0, ctxt_RB1_out, kernel_o, kernel_o2);
    timer.end();
    cout << "DONE!" << "\n";

    cout << "RB 3 ..." << endl;
    timer.start(" RB 3 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB3_out = RB(context, pack, eval, 0, ctxt_RB2_out, kernel_o, kernel_o2);
    timer.end();
    cout << "DONE!" << "\n";


    // Down Sampling (Residual) Block 1
    cout << "DSB 1 ..." << endl;
    timer.start(" DSB 1 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_DSB1_out = DSB(timer, context, pack, eval, 0, ctxt_RB3_out, kernel_o, kernel_o2, kernel_o3);
    cout << "DONE!" << "\n";


    // Residual Block 4, 5
    cout << "RB 4..." << endl;
    timer.start(" RB 4 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB4_out = RB(context, pack, eval, 1, ctxt_DSB1_out, kernel_o, kernel_o2);
    timer.end();
    cout << "DONE!" << "\n";

    cout << "RB 5 ..." << endl;
    timer.start(" RB 5 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB5_out = RB(context, pack, eval, 1, ctxt_RB4_out, kernel_o, kernel_o2);
    timer.end();
    cout << "DONE!" << "\n";


    // Down Sampling (Residual) Block 2
    cout << "DSB 2 ..." << endl;
    timer.start(" DSB 2 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_DSB2_out = DSB(timer, context, pack, eval, 1, ctxt_RB5_out, kernel_o, kernel_o2, kernel_o3);
    cout << "DONE!" << "\n";


    // Residual Block 6, 7
    cout << "RB 6..." << endl;
    timer.start(" RB 6 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB6_out = RB(context, pack, eval, 2, ctxt_DSB2_out, kernel_o, kernel_o2);
    timer.end();
    cout << "DONE!" << "\n";

    cout << "RB 7 ..." << endl;
    timer.start(" RB 7 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB7_out = RB(context, pack, eval, 2, ctxt_RB6_out, kernel_o, kernel_o2);
    timer.end();
    cout << "DONE!" << "\n";


    // Average Pooling, Flatten, FC64
    // Avg Pool
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

    // FC64
    cout << "evaluating FC64" << endl;
    timer.start("* ");
    ctxt_out = FC64old(context, pack, eval, ctxt, ptxt_vec);
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

    








    
    


    // /////////////// Decryption ////////////////
    // for (int i = 0; i < 4; ++i) {
    //     for (int ch = 0; ch < 32; ++ch) {
    //         Message dmsg;
    //         cout << "Decrypt ... ";
    //         dec.decrypt(ctxt_out[i][ch], sk, dmsg);
    //         cout << "done" << endl;
    //         // printMessage(dmsg);
    //     }
    // }

    return 0;
}
