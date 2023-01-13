////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include<iostream>
#include<thread>
#include "examples.hpp"
#include "Conv.hpp"
#include "HEaaNTimer.hpp"

// #include "HEaaN/heaan.hpp"

int main() {
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    const auto log_slots = getLogFullSlots(context);

    // Generate a new secret key
    HEaaN::SecretKey sk(context);

    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... ";
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl;

    std::cout << "Generate multiplication key ... ";
    keygen.genMultiplicationKey();
    std::cout << "done" << std::endl;

    std::cout << "Generate rotation key ... ";
    keygen.genRotationKeyBundle();
    std::cout << "done" << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);
    HEaaN::HomEvaluator eval(context, pack);
    // HEaaN::HEaaNTimer timer(false);






    ///////////// Message ///////////////////
    HEaaN::Message msg(log_slots);
    // fillRandomComplex(msg);
    std::optional<size_t> num;
    size_t length = num.has_value() ? num.value() : msg.getSize();
    size_t idx = 0;
    for (; idx < length; ++idx) {
        msg[idx].real(1.0);
        msg[idx].imag(0.0);
    }
    // If num is less than the size of msg,
    // all remaining slots are zero.
    for (; idx < msg.getSize(); ++idx) {
        msg[idx].real(0.0);
        msg[idx].imag(0.0);
    }
    printMessage(msg);

    // Ciphertext
    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... ";
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;



    std::vector<HEaaN::Ciphertext> ctxt_i;
    for (int i = 0; i < 16; ++i) {   // input channel 16
        ctxt_i.push_back(ctxt);
    }


    // std::cout << ctxt.getSize() << "\n";


    /////////////// Kernel ////////////////
    
    std::cout << "Kernel generation ... " << std::endl;
    HEaaN::EnDecoder ecd(context);
    HEaaN::Plaintext ptxt(context);
    ptxt = ecd.encode(msg, 12, 0);


    std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_o;
    for (int o = 0; o < 32; ++o) { //32
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
    std::cout << "done" << std::endl;

    // std::cout << std::thread::hardware_concurrency() << "\n";
    HEaaN::HEaaNTimer timer(false);
    timer.start("* ");
    std::vector<HEaaN::Ciphertext> ctxt_out_bundle;
    ctxt_out_bundle = Conv(context, pack, eval, 32, 2, 2, 16, 32, ctxt_i, kernel_o);
    timer.end();

    





    // TEST //
    HEaaN::Ciphertext ctxt_test(context);








    /////////////// Decryption ////////////////
    HEaaN::Message dmsg;

    for (int i = 0; i < ctxt_out_bundle.size(); ++i) {
        // std::cout << "Decrypt ... ";
        dec.decrypt(ctxt_out_bundle[i], sk, dmsg);
        // std::cout << "done" << std::endl;
        // printMessage(dmsg);
    }

    return 0;
}
