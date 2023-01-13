////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "HEaaN/heaan.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "DSB_test.hpp"
#include <stdio.h>
#include <omp.h>
#include <optional>
#include <vector>

namespace {
	using namespace HEaaN;
	using namespace std;
}


int main() {
    // Multi-thread test
    //#pragma omp parallel for
    //for (int i = 0; i < 5; ++i) {
    //    printf("thread %d : %d (in parallel)\n", omp_get_thread_num(), i);
    //}




    HEaaNTimer timer(false);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    ParameterPreset preset = ParameterPreset::FGb;
    Context context = makeContext(preset);

    //if (!isBootstrappableParameter(context)) {
    //    std::cout << "Bootstrap is not available for parameter "
    //        << presetNamer(preset) << std::endl;
    //    return -1;
    //}

    //std::cout << "Parameter : " << presetNamer(preset) << std::endl
    //   << std::endl;

    const auto log_slots = getLogFullSlots(context);

    SecretKey sk(context);
    KeyPack pack(context);
    KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... " << std::endl;
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl << std::endl;

    /*
    You should perform makeBootstrappble function
    before generating evaluation keys and constucting HomEvaluator class.
    */
    makeBootstrappable(context);

    std::cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << std::endl;
    keygen.genCommonKeys();
    std::cout << "done" << std::endl << std::endl;

    Encryptor enc(context);
    Decryptor dec(context);

    /*
    HomEvaluator constructor pre-compute the constants for bootstrapping.
    */
    std::cout << "Generate HomEvaluator (including pre-computing constants for "
        "bootstrapping) ..."
        << std::endl;
    timer.start("* ");
    HomEvaluator eval(context, pack);
    timer.end();




    ///////////// Message ///////////////////
    Message msg(log_slots);
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

    //printMessage(msg);

    Ciphertext ctxt(context);
    std::cout << "Encrypt ... " << std::endl;
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;


    /////////////// Kernel ////////////////
    // timer.start(" Kernel generation ");
    std::cout << "Kernel generation ... " << std::endl;
    //timer.start("* ");
    EnDecoder ecd(context);
    Plaintext ptxt(context);
    ptxt = ecd.encode(msg, 12, 0);

    std::vector<std::vector<std::vector<Plaintext>>> kernel_o(32);
    #pragma omp parallel
    #pragma omp for
    for (int o = 0; o < 32; ++o) { // output channel 32
        std::vector<std::vector<Plaintext>> kernel_i(16);
        for (int i = 0; i < 16; ++i) {   // input channel 16
            std::vector<Plaintext> kernel_bundle;
            Message kernel_msg(log_slots);
            Plaintext kernel(context);
            for (int k = 0; k < 9; ++k) {
                int idx = 0; //int가 없어서 오류.
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
            kernel_i[i] = kernel_bundle;
        }
        kernel_o[o] = kernel_i;
        //printf("thread %d : %d (in parallel)\n", omp_get_thread_num(), o);
    }
    //timer.end();

    

    std::vector<std::vector<std::vector<Plaintext>>> kernel_o2(32);
    #pragma omp parallel //추가 
    #pragma omp for //추가
    for (int o = 0; o < 32; ++o) { // output channel 32
        std::vector<std::vector<Plaintext>> kernel_i2(32);
        for (int i = 0; i < 32; ++i) {   // input channel 32
            std::vector<Plaintext> kernel_bundle2;
            Message kernel_msg2(log_slots);
            Plaintext kernel2(context);
            for (int k = 0; k < 9; ++k) {
                int idx = 0;
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
            kernel_i2[i] = kernel_bundle2;
        }
        kernel_o2[o] = kernel_i2;
        //printf("thread %d : %d (in parallel)\n", omp_get_thread_num(), o);
    }
    
    std::vector<std::vector<std::vector<Plaintext>>> kernel_o3(32);
    #pragma omp parallel //추가 
    #pragma omp for //추가
    for (int o = 0; o < 32; ++o) { // output channel 32
        std::vector<std::vector<Plaintext>> kernel_i3(16);
        for (int i = 0; i < 16; ++i) {   // input channel 16
            std::vector<Plaintext> kernel_bundle3;
            Message kernel_msg3(log_slots);
            Plaintext kernel3(context);
            for (int k = 0; k < 1; ++k) {
                int idx = 0;
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
            kernel_i3[i] = kernel_bundle3;
        }
        kernel_o3[o] = kernel_i3;
        //printf("thread %d : %d (in parallel)\n", omp_get_thread_num(), o);
    }
    std::cout << "done" << std::endl;




    std::vector<std::vector<Ciphertext>> ctxt_bundle;
    // #pragma omp parallel for
    for (int i = 0; i < 16; ++i) {
        std::vector<Ciphertext> ctxt_bundle_cache;
        for (int ch = 0; ch < 16; ++ch) {
            ctxt_bundle_cache.push_back(ctxt);
        }
        ctxt_bundle.push_back(ctxt_bundle_cache);
    }
    std::cout << "SETUP is over" << "\n";

    timer.start(" DSB ");
    std::vector<std::vector<Ciphertext>> ctxt_out;
    ctxt_out = DSB(context, pack, eval, 0, ctxt_bundle, kernel_o, kernel_o2, kernel_o3);
    timer.end();
    std::cout << "DSB is over" << "\n";








    
    


    // /////////////// Decryption ////////////////
    // for (int i = 0; i < 4; ++i) {
    //     for (int ch = 0; ch < 32; ++ch) {
    //         Message dmsg;
    //         std::cout << "Decrypt ... ";
    //         dec.decrypt(ctxt_out[i][ch], sk, dmsg);
    //         std::cout << "done" << std::endl;
    //         // printMessage(dmsg);
    //     }
    // }

    

    return 0;
}
