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

HEaaN::Ciphertext Conv(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, HEaaN::Ciphertext ctxt, HEaaN::Message kernel) {
    // Make rotated ctxts
    std::vector<HEaaN::Ciphertext> rotated_ctxts;
    HEaaN::Ciphertext rotated_ctxts_cache(context);
    eval.leftRotate(ctxt, -(imgsize + (gap * stride)), rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, -imgsize, rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, -(imgsize-(gap * stride)), rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, -(gap * stride), rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, 0, rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, (gap * stride), rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, imgsize-(gap * stride), rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, imgsize, rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    eval.leftRotate(ctxt, imgsize+(gap * stride), rotated_ctxts_cache);
    rotated_ctxts.push_back(rotated_ctxts_cache);
    
    // Convolution
    HEaaN::Ciphertext ctxt_out(context);
    eval.mult(rotated_ctxts[0], kernel, rotated_ctxts[0]);
    ctxt_out = rotated_ctxts[0];
    for (int i = 1; i < 9; i++) {
        eval.mult(rotated_ctxts[i], kernel, rotated_ctxts[i]);
        eval.add(ctxt_out, rotated_ctxts[i], ctxt_out);
    }
    return ctxt_out;
}

int main() {
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    const auto log_slots = getLogFullSlots(context);

    // Generate a new secret key
    HEaaN::SecretKey sk(context);

    /*
    You can also use the constuctors
    SecretKey(const Context &context, std::istream &stream) or
    SecretKey(const Context &context, const std::string &key_dir_path)
    if you have the saved secret key file.
    */

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

    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... ";
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;

    /////////////// Kernel ////////////////
    HEaaN::Message kernel(log_slots);
    idx = 0;
    for (; idx < length; ++idx) {
        kernel[idx].real(2.0);
        kernel[idx].imag(0.0);
    }
    // If num is less than the size of msg,
    // all remaining slots are zero.
    for (; idx < kernel.getSize(); ++idx) {
        kernel[idx].real(0.0);
        kernel[idx].imag(0.0);
    }



    HEaaN::Ciphertext ctxt_out(context);
    ctxt_out = Conv(context, pack, eval, 32, 2, 1, ctxt, kernel);




    /////////////// Decryption ////////////////
    HEaaN::Message dmsg;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_out, sk, dmsg);
    std::cout << "done" << std::endl;
    printMessage(dmsg);

    return 0;
}
