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

    printMessage(msg);

    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... ";
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;

    /////////////// Kernel ////////////////
    std::vector<HEaaN::Message> kernel_bundle;
    for (int i = 0; i < 9; ++i) {
        HEaaN::Message kernel(log_slots);
        idx = 0;
        for (; idx < length; ++idx) {
            kernel[idx].real(0.1);
            kernel[idx].imag(0.0);
        }
        // If num is less than the size of msg,
        // all remaining slots are zero.
        for (; idx < kernel.getSize(); ++idx) {
            kernel[idx].real(0.0);
            kernel[idx].imag(0.0);
        }
        kernel_bundle.push_back(kernel);
    }

    HEaaN::Ciphertext ctxt0(context);
    HEaaN::Ciphertext ctxt1(context);
    HEaaN::Ciphertext ctxt2(context);
    HEaaN::Ciphertext ctxt3(context); 
    ctxt0 = ctxt;
    ctxt1 = ctxt;
    ctxt2 = ctxt;
    ctxt3 = ctxt;










    
    std::vector<HEaaN::Ciphertext> ctxt_bundle;
    ctxt_bundle.push_back(ctxt0);
    ctxt_bundle.push_back(ctxt1);
    ctxt_bundle.push_back(ctxt2);
    ctxt_bundle.push_back(ctxt3);

    // Main flow
    std::vector<HEaaN::Ciphertext> ctxt_conv_out_bundle;
    HEaaN::Ciphertext ctxt_conv_out_cache(context);
    for (int i = 0; i < 4; ++i) {
        ctxt_conv_out_cache = Conv(context, pack, eval, 32, 2, 1, ctxt_bundle[i], kernel_bundle);
        ctxt_conv_out_bundle.push_back(ctxt_conv_out_cache);
    }
    
    HEaaN::Ciphertext ctxt_MPP_out(context);
    ctxt_MPP_out = MPPacking(context, pack, eval, 32, ctxt_conv_out_bundle);

    /////////////// Decryption ////////////////
    HEaaN::Message dmsg0;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_MPP_out, sk, dmsg0);
    std::cout << "done" << std::endl;
    printMessage(dmsg0);
    ////////////////////////////////////////

    HEaaN::Ciphertext ctxt_relu_out(context);
    ApproxReLU(context, eval, ctxt_MPP_out, ctxt_relu_out);

    HEaaN::Ciphertext ctxt_conv_out2(context);
    ctxt_conv_out2 = Conv(context, pack, eval, 32, 1, 1, ctxt_relu_out, kernel_bundle);

    


    // Residual flow
    std::vector<HEaaN::Ciphertext> ctxt_residual_out;
    HEaaN::Ciphertext ctxt_residual_out_cache(context);
    for (int i = 0; i < 4; ++i) {
        ctxt_residual_out_cache = Conv(context, pack, eval, 32, 2, 1, ctxt_bundle[i], kernel_bundle);
        ctxt_residual_out.push_back(ctxt_residual_out_cache);
    }

    HEaaN::Ciphertext ctxt_residual_MPP_out(context);
    ctxt_residual_MPP_out = MPPacking(context, pack, eval, 32, ctxt_residual_out);



    // Main flow + Residual flow
    HEaaN::Ciphertext ctxt_residual_added(context);
    eval.add(ctxt_conv_out2, ctxt_residual_MPP_out, ctxt_residual_added);

    HEaaN::Ciphertext ctxt_out(context);
    ApproxReLU(context, eval, ctxt_residual_added, ctxt_out);





    /////////////// Decryption ////////////////
    HEaaN::Message dmsg;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_out, sk, dmsg);
    std::cout << "done" << std::endl;
    printMessage(dmsg);

    return 0;
}
