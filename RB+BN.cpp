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
#include "RB+BN.hpp"
#include "kernelEncode.hpp"


namespace {
    using namespace HEaaN;
    using namespace std;
}

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

    EnDecoder ecd(context);
    // ///////////// Preset ///////////////////
    // std::int rb_num;
    // rb_num = 16;
    
    Ciphertext ctxt(context);
    vector<vector<Ciphertext>> ctxt_relu1_out_bundle(16,vector<Ciphertext>(16,ctxt));
    Plaintext ptxt_init(context);
    // Residual Block 1, 2, 3
        // RB 1 - 1
    std::cout << "data load... " << std::endl;
    vector<double> temp1;
    vector<vector<vector<Plaintext>>> block1conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path1 = "/app/examples/kernel/multiplicands/" + string("block1conv0multiplicands16_16_3_3");
    txtreader(temp1, path1);
    kernel_ptxt(context, temp1, block1conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);
    
    vector<double> block1conv0summands16;
    string path1a = "/app/examples/kernel/summands/" + string("block1conv0summands16");
    txtreader(block1conv0summands16, path1a);
    
    for(int i = 0 ; i < block1conv0multiplicands16_16_3_3.size() ;i++){
        std::cout << i << " th level is " << block1conv0multiplicands16_16_3_3[0][i].getLevel() << std::endl;
    }   
    
        // RB 1 - 2
    vector<double> temp2;
    vector<vector<vector<Plaintext>>> block1conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path2 = "/app/examples/kernel/multiplicands/" + string("block1conv1multiplicands16_16_3_3");
    txtreader(temp2, path2);
    kernel_ptxt(context, temp2, block1conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block1conv1summands16;
    string path2a = "/app/examples/kernel/summands/" + string("block1conv1summands16");
    txtreader(block1conv1summands16, path2a);
    std::cout << "done" << std::endl;
    
    for(int i = 0 ; i < block1conv1multiplicands16_16_3_3.size() ;i++){
        std::cout << i << " th level is " << block1conv1multiplicands16_16_3_3[0][i].getLevel() << std::endl;
    }   



    // // RB 1
    // cout << "RB 1 ..." << endl;
    // timer.start(" RB 1 ");
    // vector<vector<Ciphertext>> ctxt_RB1_out;
    // ctxt_RB1_out = RB(context, pack, eval, 0, ctxt_relu1_out_bundle, block1conv0multiplicands16_16_3_3, block1conv1multiplicands16_16_3_3,
    // block1conv0summands16, block1conv1summands16);
    // timer.end();
    // cout << "DONE!" << "\n";
    



    /*

    ///////////// Message & Ctxt ///////////////////
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

    printMessage(msg);

    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... " << std::endl;
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;

    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_bundle;
    for (int i = 0; i < 16; ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_bundle_cache;
        for (int ch = 0; ch < 16; ++ch) {
            ctxt_bundle_cache.push_back(ctxt);
        }
        ctxt_bundle.push_back(ctxt_bundle_cache);
    }


    /////////////// Kernel ////////////////
    // timer.start(" Kernel generation ");
    std::cout << "Kernel generation ... " << std::endl;

    HEaaN::EnDecoder ecd(context);
    HEaaN::Plaintext ptxt(context);
    ptxt = ecd.encode(msg, 12, 0);

    std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_o;
    for (int o = 0; o < 16; ++o) { // output channel 16
        std::vector<std::vector<HEaaN::Plaintext>> kernel_i;
        for (int i = 0; i < 16; ++i) {   // input channel 16
            std::vector<HEaaN::Plaintext> kernel_bundle;
            HEaaN::Message kernel_msg(log_slots);
            HEaaN::Plaintext kernel(context);
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
    for (int o = 0; o < 16; ++o) { // output channel 16
        std::vector<std::vector<HEaaN::Plaintext>> kernel_i2;
        for (int i = 0; i < 16; ++i) {   // input channel 16
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
    

    // std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_o3;
    // for (int o = 0; o < 32; ++o) { // output channel 32
    //     std::vector<std::vector<HEaaN::Plaintext>> kernel_i3;
    //     for (int i = 0; i < 16; ++i) {   // input channel 16
    //         std::vector<HEaaN::Plaintext> kernel_bundle3;
    //         HEaaN::Message kernel_msg3(log_slots);
    //         HEaaN::Plaintext kernel3(context);
    //         for (int k = 0; k < 1; ++k) {
    //             idx = 0;
    //             for (; idx < length; ++idx) {
    //                 kernel_msg3[idx].real(2.0);
    //                 kernel_msg3[idx].imag(0.0);
    //             }
    //             // If num is less than the size of msg,
    //             // all remaining slots are zero.
    //             for (; idx < kernel_msg3.getSize(); ++idx) {
    //                 kernel_msg3[idx].real(0.0);
    //                 kernel_msg3[idx].imag(0.0);
    //             }
    //             kernel3 = ecd.encode(kernel_msg3, 12, 0);
    //             kernel_bundle3.push_back(kernel3);
    //         }
    //         kernel_i3.push_back(kernel_bundle3);
    //     }
    //     kernel_o3.push_back(kernel_i3);
    // }
    std::cout << "done" << std::endl;
    // timer.end();






    std::cout << "SETUP is over" << "\n";

    // int num_ctxt;
    // num_ctxt = ctxt_bundle.size();

    // int num_kernel_bundle1;
    // num_kernel_bundle1 = kernel_o.size();

    // int num_kernel_bundle2;
    // num_kernel_bundle2 = kernel_o2.size();


    timer.start(" RB ");
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_out;
    ctxt_out = RB(context, pack, eval, 0, ctxt_bundle, kernel_o, kernel_o2);
    timer.end();
    std::cout << "RB is over" << "\n";








    
    


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
    */

    return 0;
}
