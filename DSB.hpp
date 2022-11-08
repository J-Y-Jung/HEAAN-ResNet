////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


HEaaN::Ciphertext DSB(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, std::vector<HEaaN::Ciphertext> ctxt_bundle, 
std::vector<HEaaN::Message> kernel_bundle, std::vector<HEaaN::Message> kernel_bundle2, 
std::vector<HEaaN::Message> kernel_residual_bundle) {

    // Main flow
    std::vector<HEaaN::Ciphertext> ctxt_conv_out_bundle;
    HEaaN::Ciphertext ctxt_conv_out_cache(context);
    for (int i = 0; i < 4; ++i) {
        ctxt_conv_out_cache = Conv(context, pack, eval, 32, 2, 1, ctxt_bundle[i], kernel_bundle);
        ctxt_conv_out_bundle.push_back(ctxt_conv_out_cache);
    }
    
    HEaaN::Ciphertext ctxt_MPP_out(context);
    ctxt_MPP_out = MPPacking(context, pack, eval, 32, ctxt_conv_out_bundle);

    // /////////////// Decryption ////////////////
    // HEaaN::Message dmsg0;
    // std::cout << "Decrypt ... ";
    // dec.decrypt(ctxt_MPP_out, sk, dmsg0);
    // std::cout << "done" << std::endl;
    // printMessage(dmsg0);
    // ////////////////////////////////////////

    HEaaN::Ciphertext ctxt_relu_out(context);
    ApproxReLU(context, eval, ctxt_MPP_out, ctxt_relu_out);

    HEaaN::Ciphertext ctxt_conv_out2(context);
    ctxt_conv_out2 = Conv(context, pack, eval, 32, 1, 1, ctxt_relu_out, kernel_bundle2);

    


    // Residual flow
    std::vector<HEaaN::Ciphertext> ctxt_residual_out;
    HEaaN::Ciphertext ctxt_residual_out_cache(context);
    for (int i = 0; i < 4; ++i) {
        ctxt_residual_out_cache = Conv(context, pack, eval, 32, 2, 1, ctxt_bundle[i], kernel_residual_bundle);
        ctxt_residual_out.push_back(ctxt_residual_out_cache);
    }

    HEaaN::Ciphertext ctxt_residual_MPP_out(context);
    ctxt_residual_MPP_out = MPPacking(context, pack, eval, 32, ctxt_residual_out);



    // Main flow + Residual flow
    HEaaN::Ciphertext ctxt_residual_added(context);
    eval.add(ctxt_conv_out2, ctxt_residual_MPP_out, ctxt_residual_added);

    HEaaN::Ciphertext ctxt_out(context);
    ApproxReLU(context, eval, ctxt_residual_added, ctxt_out);


    return ctxt_out;
}
