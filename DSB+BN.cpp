////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "examples.hpp"
#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "DSB+BN.hpp"
#include "cifar-10-reader/include/cifar/cifar10_reader.hpp"
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

namespace {
using namespace HEaaN;
using namespace std;
}


int main() {
    // Multi-thread test
    #pragma omp parallel for
    for (int i = 0; i < 5; ++i) {
        printf("thread %d : %d (in parallel)\n", omp_get_thread_num(), i);
    }





    // // bin 폴더 안에 bin 데이터셋 파일 있어야 함
    // auto dataset = cifar::read_dataset<vector, vector, uint8_t, uint8_t>();
    // cout << isprint(dataset.test_images[16][16]) << "\n";

    // int imgpix;
    // int imgpix2;
    // imgpix = (char)dataset.test_images[16][16];
    // printf("%d\n",imgpix);
    // imgpix2 = (double)dataset.test_images[16][16];
    // cout << imgpix2 << "\n";



















    HEaaNTimer timer(false);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
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

    /*
    You should perform makeBootstrappble function
    before generating evaluation keys and constucting HomEvaluator class.
    */
    makeBootstrappable(context);

    cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << endl;
    keygen.genCommonKeys();
    cout << "done" << endl << endl;

    Encryptor enc(context);
    Decryptor dec(context);

    /*
    HomEvaluator constructor pre-compute the constants for bootstrapping.
    */
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

    printMessage(msg);

    Ciphertext ctxt(context);
    cout << "Encrypt ... " << endl;
    enc.encrypt(msg, pack, ctxt); // public key encryption
    cout << "done" << endl;
    cout << "level of ctxt is " << ctxt.getLevel() << "\n";
    ctxt.setLevel(6);
    cout << "level of ctxt(After rescale) is " << ctxt.getLevel() << "\n";

    /////////////// Kernel ////////////////
    // timer.start(" Kernel generation ");
    cout << "Kernel generation ... " << endl;
    timer.start("* ");
    EnDecoder ecd(context);
    // Plaintext ptxt(context);
    // ptxt = ecd.encode(msg, 12, 0);

    vector<vector<vector<Plaintext>>> kernel_o(32);
    #pragma omp parallel
    #pragma omp for
    for (int o = 0; o < 32; ++o) { // output channel 32
        vector<vector<Plaintext>> kernel_i(16);
        for (int i = 0; i < 16; ++i) {   // input channel 16
            vector<Plaintext> kernel_bundle;
            Message kernel_msg(log_slots);
            Plaintext kernel(context);
            for (int k = 0; k < 9; ++k) {
                for (int idx = 0; idx < length; ++idx) {
                    kernel_msg[idx].real(2.0);
                    kernel_msg[idx].imag(0.0);
                }
                // If num is less than the size of msg,
                // all remaining slots are zero.
                for (; idx < kernel_msg.getSize(); ++idx) {
                    kernel_msg[idx].real(0.0);
                    kernel_msg[idx].imag(0.0);
                }
                kernel = ecd.encode(kernel_msg, 6, 0);
                kernel_bundle.push_back(kernel);
            }
            kernel_i[i] = kernel_bundle;
        }
        kernel_o[o] = kernel_i;
    }
    // vector<vector<vector<Plaintext>>>().swap(kernel_o);
    
    

    vector<vector<vector<Plaintext>>> kernel_o2(32);
    #pragma omp parallel
    #pragma omp for
    for (int o = 0; o < 32; ++o) { // output channel 32
        vector<vector<Plaintext>> kernel_i2(32);
        for (int i = 0; i < 32; ++i) {   // input channel 32
            vector<Plaintext> kernel_bundle2;
            Message kernel_msg2(log_slots);
            Plaintext kernel2(context);
            for (int k = 0; k < 9; ++k) {
                for (int idx = 0; idx < length; ++idx) {
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
    }
    // vector<vector<vector<Plaintext>>>().swap(kernel_o2);

    vector<vector<vector<Plaintext>>> kernel_o3(32);
    #pragma omp parallel
    #pragma omp for
    for (int o = 0; o < 32; ++o) { // output channel 32
        vector<vector<Plaintext>> kernel_i3(16);
        for (int i = 0; i < 16; ++i) {   // input channel 16
            vector<Plaintext> kernel_bundle3;
            Message kernel_msg3(log_slots);
            Plaintext kernel3(context);
            for (int k = 0; k < 1; ++k) {
                for (int idx = 0; idx < length; ++idx) {
                    kernel_msg3[idx].real(2.0);
                    kernel_msg3[idx].imag(0.0);
                }
                // If num is less than the size of msg,
                // all remaining slots are zero.
                for (; idx < kernel_msg3.getSize(); ++idx) {
                    kernel_msg3[idx].real(0.0);
                    kernel_msg3[idx].imag(0.0);
                }
                kernel3 = ecd.encode(kernel_msg3, 6, 0);
                kernel_bundle3.push_back(kernel3);
            }
            kernel_i3[i] = kernel_bundle3;
        }
        kernel_o3[o] = kernel_i3;
    }
    // vector<vector<vector<Plaintext>>>().swap(kernel_o3);
    cout << "done" << endl;
    timer.end();
    // cout << "ctxt is " << sizeof(ctxt) << " bytes" << endl;


    // Ctxt bundle
    vector<vector<Ciphertext>> ctxt_bundle(16);
    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i < 16; ++i) {
        vector<Ciphertext> ctxt_bundle_cache;
        for (int ch = 0; ch < 16; ++ch) {
            ctxt_bundle_cache.push_back(ctxt);
        }
        ctxt_bundle[i] = ctxt_bundle_cache;
    }

    // BN sum ptxts

    Plaintext ptxt(context);
    ptxt = ecd.encode(msg, 4, 0);


    vector<double> BN1_add;
    for (int ch = 0; ch < 32; ++ch) {
        // BN1_add.push_back(ptxt);
    }

    vector<double> BN2_add;
    for (int ch = 0; ch < 32; ++ch) {
        // BN2_add.push_back(ptxt);
    }



    cout << "SETUP is over" << "\n";


    timer.start(" DSB ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_out = DSB(context, pack, eval, 0, ctxt_bundle,
        kernel_o, kernel_o2, kernel_o3, BN1_add, BN2_add);
    timer.end();
    cout << "DSB is over" << "\n";










    
    


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
