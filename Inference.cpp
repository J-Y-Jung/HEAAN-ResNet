#include "examples.hpp"
#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "DSB.hpp"

int main() {
    // SetUp
    HEaaN::HEaaNTimer timer(false);
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

    HEaaN::makeBootstrappable(context);

    std::cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << std::endl;
    keygen.genCommonKeys();
    std::cout << "done" << std::endl << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);

    std::cout << "Generate HomEvaluator (including pre-computing constants for "
        "bootstrapping) ..."
        << std::endl;
    timer.start("* ");
    HEaaN::HomEvaluator eval(context, pack);
    timer.end();




    ///////////// Message ///////////////////
    HEaaN::Message msg(log_slots);
    // fillRandomComplex(msg);
    std::optional<size_t> num;
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

    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... " << std::endl;
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;


    /////////////// Kernel ////////////////
    // timer.start(" Kernel generation ");
    std::cout << "Kernel generation ... " << std::endl;
    timer.start("* ");
    HEaaN::EnDecoder ecd(context);
    HEaaN::Plaintext ptxt(context);
    ptxt = ecd.encode(msg, 12, 0);

    std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_o;
    // #pragma omp parallel for
    for (int o = 0; o < 32; ++o) { // output channel 32
        std::vector<std::vector<HEaaN::Plaintext>> kernel_i;
        // #pragma omp parallel for
        for (int i = 0; i < 16; ++i) {   // input channel 16
            std::vector<HEaaN::Plaintext> kernel_bundle;
            HEaaN::Message kernel_msg(log_slots);
            HEaaN::Plaintext kernel(context);
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

    std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_o2;
    // #pragma omp parallel for
    for (int o = 0; o < 32; ++o) { // output channel 32
        std::vector<std::vector<HEaaN::Plaintext>> kernel_i2;
        for (int i = 0; i < 32; ++i) {   // input channel 32
            std::vector<HEaaN::Plaintext> kernel_bundle2;
            HEaaN::Message kernel_msg2(log_slots);
            HEaaN::Plaintext kernel2(context);
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

    std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_o3;
    // #pragma omp parallel for
    for (int o = 0; o < 32; ++o) { // output channel 32
        std::vector<std::vector<HEaaN::Plaintext>> kernel_i3;
        for (int i = 0; i < 16; ++i) {   // input channel 16
            std::vector<HEaaN::Plaintext> kernel_bundle3;
            HEaaN::Message kernel_msg3(log_slots);
            HEaaN::Plaintext kernel3(context);
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
    std::cout << "done" << std::endl;
    timer.end();



    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_bundle;
    // #pragma omp parallel for
    for (int i = 0; i < 16; ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_bundle_cache;
        for (int ch = 0; ch < 3; ++ch) {
            ctxt_bundle_cache.push_back(ctxt);
        }
        ctxt_bundle.push_back(ctxt_bundle_cache);
    }

    std::cout << "SETUP is over" << "\n";

    // Convolution 1
    std::cout << "Convolution 1 ..." << std::endl;
    timer.start(" Convolution 1 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_conv1_out_bundle;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        std::vector<HEaaN::Ciphertext> ctxt_conv1_out_cache;
        ctxt_conv1_out_cache = Conv(context, pack, eval, 32, 1, 1, 3, 16, ctxt_bundle[i], kernel_bundle);
        ctxt_conv1_out_bundle.push_back(ctxt_conv1_out_cache);
    }
    timer.end();
    std::cout << "DONE!" << "\n";


    // AppReLU
    std::cout << "AppReLU ..." << std::endl;
    timer.start(" AppReLU 1 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_relu1_out_bundle;
    for (int i = 0; i < 4; ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_relu1_out_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            std::cout << "(i = " << i << ", " << "ch = " << ch << ")" << "\n";
            HEaaN::Ciphertext ctxt_relu1_out(context);
            ApproxReLU(context, eval, ctxt_conv1_out_bundle[i][ch], ctxt_relu1_out);
            ctxt_relu1_out_allch_bundle.push_back(ctxt_relu1_out);
        }
        ctxt_relu1_out_bundle.push_back(ctxt_relu1_out_allch_bundle);
    }
    timer.end();
    std::cout << "DONE!" << "\n";


    // Residual Block 1, 2, 3
    std::cout << "RB 1 ..." << std::endl;
    timer.start(" RB 1 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 0, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "DONE!" << "\n";

    std::cout << "RB 2..." << std::endl;
    timer.start(" RB 2 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 0, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "DONE!" << "\n";

    std::cout << "RB 3 ..." << std::endl;
    timer.start(" RB 3 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 0, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "DONE!" << "\n";


    // Down Sampling (Residual) Block 1
    std::cout << "DSB 1 ..." << std::endl;
    timer.start(" DSB 1 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = DSB(timer, context, pack, eval, 0, ctxt_bundle, kernel_o, kernel_o2, kernel_o3);
    std::cout << "DONE!" << "\n";


    // Residual Block 4, 5
    std::cout << "RB 4..." << std::endl;
    timer.start(" RB 4 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 1, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "DONE!" << "\n";

    std::cout << "RB 5 ..." << std::endl;
    timer.start(" RB 5 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 1, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "DONE!" << "\n";


    // Down Sampling (Residual) Block 2
    std::cout << "DSB 2 ..." << std::endl;
    timer.start(" DSB 2 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = DSB(timer, context, pack, eval, 1, ctxt_bundle, kernel_o, kernel_o2, kernel_o3);
    std::cout << "DONE!" << "\n";


    // Residual Block 6, 7
    std::cout << "RB 6..." << std::endl;
    timer.start(" RB 6 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 2, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "DONE!" << "\n";

    std::cout << "RB 7 ..." << std::endl;
    timer.start(" RB 7 ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 2, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "DONE!" << "\n";


    // Average Pooling, Flatten, FC64


    








    
    


    // /////////////// Decryption ////////////////
    // for (int i = 0; i < 4; ++i) {
    //     for (int ch = 0; ch < 32; ++ch) {
    //         HEaaN::Message dmsg;
    //         std::cout << "Decrypt ... ";
    //         dec.decrypt(ctxt_out[i][ch], sk, dmsg);
    //         std::cout << "done" << std::endl;
    //         // printMessage(dmsg);
    //     }
    // }

    return 0;
}
