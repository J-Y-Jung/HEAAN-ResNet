////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<HEaaN::Ciphertext>> RB(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int DSB_count, std::vector<std::vector<HEaaN::Ciphertext>> ctxt_bundle, 

std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_bundle, 
std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_bundle2) {
    ///////////////////////// SetUp ////////////////////////////////
    std::cout << "RB start" << "\n";
    // int num_ctxt;
    // num_ctxt = ctxt_bundle.size();

    // int num_kernel_bundle1;
    // num_kernel_bundle1 = kernel_bundle.size();

    // int num_kernel_bundle2;
    // num_kernel_bundle2 = kernel_bundle2.size();


    ///////////////////////// Main flow /////////////////////////////////////////
    std::cout << "First Conv-(main flow) ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_conv_out_bundle;
    for (int i = 0; i < 16; ++i) { 
        std::vector<HEaaN::Ciphertext> ctxt_conv_out_cache;
        ctxt_conv_out_cache = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_bundle[i], kernel_bundle);
        ctxt_conv_out_bundle.push_back(ctxt_conv_out_cache);
    }
    std::cout << "DONE!" << "\n";
    
    // MPP input bundle making
    // std::cout << "MPP-(main flow, First Conv) ..." << std::endl;
    // std::vector<std::vector<std::vector<HEaaN::Ciphertext>>> ctxt_MPP_in;
    // for (int i = 0; i < 4; ++i) {
    //     std::vector<std::vector<HEaaN::Ciphertext>> ctxt_MPP_in_allch_bundle;
    //     for (int ch = 0; ch < 32; ++ch) {
    //         std::vector<HEaaN::Ciphertext> ctxt_MPP_in_cache;
    //         ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+0][ch]);
    //         ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+1][ch]);
    //         ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+2][ch]);
    //         ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+3][ch]);
    //         ctxt_MPP_in_allch_bundle.push_back(ctxt_MPP_in_cache);
    //     }
    //     ctxt_MPP_in.push_back(ctxt_MPP_in_allch_bundle);
    // }
    // 
    // // MPP
    // std::vector<std::vector<HEaaN::Ciphertext>> ctxt_MPP_out_bundle;
    // for (int i = 0; i < 4; ++i) {
    //     std::vector<HEaaN::Ciphertext> ctxt_MPP_out;
    //     for (int ch = 0; ch < 32; ++ch) {
    //         HEaaN::Ciphertext ctxt_MPP_out_cache(context);
    //         ctxt_MPP_out_cache = MPPacking(context, pack, eval, 32, ctxt_MPP_in[i][ch]);
    //         ctxt_MPP_out.push_back(ctxt_MPP_out_cache);
    //     }
    //     ctxt_MPP_out_bundle.push_back(ctxt_MPP_out);
    // }
    // std::cout << "DONE!" << "\n";
    // ctxt_MPP_out_bundle

    // /////////////// Decryption ////////////////
    // HEaaN::Message dmsg0;
    // std::cout << "Decrypt ... ";
    // dec.decrypt(ctxt_MPP_out, sk, dmsg0);
    // std::cout << "done" << std::endl;
    // printMessage(dmsg0);
    // ////////////////////////////////////////

    // AppReLU
    std::cout << "AppReLU-(main flow) ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_relu_out_bundle;
    for (int i = 0; i < 16; ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_relu_out_allch_bundle;
        for (int ch = 0; ch < 16; ++ch) {
            std::cout << ch << "\n";
            HEaaN::Ciphertext ctxt_relu_out(context);
            ApproxReLU(context, eval, ctxt_conv_out_bundle[i][ch], ctxt_relu_out);
            ctxt_relu_out_allch_bundle.push_back(ctxt_relu_out);
        }
        ctxt_relu_out_bundle.push_back(ctxt_relu_out_allch_bundle);
    }
    std::cout << "DONE!" << "\n";

    // Second convolution
    std::cout << "Second Conv-(main flow) ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_conv_out2_bundle;
    for (int i = 0; i < 16; ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_conv_out2_allch_bundle;
        ctxt_conv_out2_allch_bundle = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_relu_out_bundle[i], kernel_bundle2);
        ctxt_conv_out2_bundle.push_back(ctxt_conv_out2_allch_bundle);
    }
    std::cout << "DONE!" << "\n";

    ///////////////////// Residual flow ////////////////////////////
    // // Convolution
    // std::cout << "Residual Conv-(residual flow) ..." << std::endl;
    // std::vector<std::vector<HEaaN::Ciphertext>> ctxt_residual_out_bundle;
    // for (int i = 0; i < 16; ++i) { 
    //     std::vector<HEaaN::Ciphertext> ctxt_residual_out_cache;
    //     ctxt_residual_out_cache = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_bundle[i], kernel_residual_bundle);
    //     
    //     ctxt_residual_out_bundle.push_back(ctxt_residual_out_cache);
    // }
    // std::cout << "DONE!" << "\n";
    // // MPP input bundle making
    // std::cout << "MPP-(residual flow) ..." << std::endl;
    // std::vector<std::vector<std::vector<HEaaN::Ciphertext>>> ctxt_MPP_in2;
    // for (int i = 0; i < 4; ++i) {
    //     std::vector<std::vector<HEaaN::Ciphertext>> ctxt_MPP_in_allch_bundle2;
    //     for (int ch = 0; ch < 32; ++ch) {
    //         std::vector<HEaaN::Ciphertext> ctxt_MPP_in_cache2;
    //         ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+0][ch]);
    //         ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+1][ch]);
    //         ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+2][ch]);
    //         ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+3][ch]);
    //         ctxt_MPP_in_allch_bundle2.push_back(ctxt_MPP_in_cache2);
    //     }
    //     ctxt_MPP_in2.push_back(ctxt_MPP_in_allch_bundle2);
    // }
    // // MPP
    // std::vector<std::vector<HEaaN::Ciphertext>> ctxt_MPP_out_bundle2;
    // for (int i = 0; i < 4; ++i) {
    //     std::vector<HEaaN::Ciphertext> ctxt_MPP_out2;
    //     for (int ch = 0; ch < 32; ++ch) {
    //         HEaaN::Ciphertext ctxt_MPP_out_cache2(context);
    //         ctxt_MPP_out_cache2 = MPPacking(context, pack, eval, 32, ctxt_MPP_in2[i][ch]);
    //         ctxt_MPP_out2.push_back(ctxt_MPP_out_cache2);
    //     }
    //     ctxt_MPP_out_bundle2.push_back(ctxt_MPP_out2);
    // }
    // std::cout << "DONE!" << "\n";


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    std::cout << "Main flow + Residual flow ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_residual_added;
    for (int i = 0; i < 16; ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_residual_added_allch_bundle;
        for (int ch = 0; ch < 16; ++ch) {
            HEaaN::Ciphertext ctxt_residual_added_cache(context);
            eval.add(ctxt_conv_out2_bundle[i][ch], ctxt_bundle[i][ch], ctxt_residual_added_cache);
            ctxt_residual_added_allch_bundle.push_back(ctxt_residual_added_cache);
        }
        ctxt_residual_added.push_back(ctxt_residual_added_allch_bundle);
    }
    std::cout << "DONE!" << "\n";

    // Last AppReLU
    std::cout << "Last AppReLU ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_RB_out;
    for (int i = 0; i < 16; ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_RB_out_allch_bundle;
        for (int ch = 0; ch < 16; ++ch) {
            HEaaN::Ciphertext ctxt_RB_out_cache(context);
            ApproxReLU(context, eval, ctxt_residual_added[i][ch], ctxt_RB_out_cache);
            ctxt_RB_out_allch_bundle.push_back(ctxt_RB_out_cache);
        }
        ctxt_RB_out.push_back(ctxt_RB_out_allch_bundle);
    }
    std::cout << "DONE!" << "\n";


    return ctxt_RB_out;
}
