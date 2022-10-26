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
#include "MPPacking.hpp"

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
    // Message 1
    HEaaN::Message msg(log_slots);
    // fillRandomComplex(msg);
    std::optional<size_t> num;
    // size_t length = num.has_value() ? num.value() : msg.getSize();
    size_t idx = 0;
    for (; idx < msg.getSize(); ++idx) {
        msg[idx].real(1.0);
        msg[idx].imag(0.0);
    }
    // // If num is less than the size of msg,
    // // all remaining slots are zero.
    // for (; idx < msg.getSize(); ++idx) {
    //     msg[idx].real(0.0);
    //     msg[idx].imag(0.0);
    // }

    printMessage(msg);

    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... ";
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;

    // Message 2
    HEaaN::Message msg2(log_slots);
    // fillRandomComplex(msg2);
    size_t idx2 = 0;
    for (; idx2 < msg2.getSize(); ++idx2) {
        msg2[idx2].real(7.0);
        msg2[idx2].imag(0.0);
    }
    // // If num is less than the size of msg,
    // // all remaining slots are zero.
    // for (; idx2 < msg2.getSize(); ++idx2) {
    //     msg2[idx2].real(0.0);
    //     msg2[idx2].imag(0.0);
    // }

    printMessage(msg2);

    HEaaN::Ciphertext ctxt22(context);
    std::cout << "Encrypt ... ";
    enc.encrypt(msg2, pack, ctxt22); // public key encryption
    std::cout << "done" << std::endl;


    // Save as ctxt bundle
    std::vector<HEaaN::Ciphertext> ctxt_bundle;
    ctxt_bundle.push_back(ctxt);
    ctxt_bundle.push_back(ctxt22);
    ctxt_bundle.push_back(ctxt);
    ctxt_bundle.push_back(ctxt);

    std::vector<HEaaN::Ciphertext> ctxt_bundle2;
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt22);
    ctxt_bundle2.push_back(ctxt22);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt22);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt22);
    ctxt_bundle2.push_back(ctxt22);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt);
    ctxt_bundle2.push_back(ctxt22);
    ctxt_bundle2.push_back(ctxt22);



    // MPPacking
    HEaaN::Ciphertext ctxt_out(context);
    ctxt_out = MPPacking(context, pack, eval, 32, ctxt_bundle);






    /////////////// Decryption ////////////////
    HEaaN::Message dmsg;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_out, sk, dmsg);
    std::cout << "done" << std::endl;
    printMessage(dmsg);

    return 0;
}
