////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                     //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


std::vector<HEaaN::Ciphertext> Conv(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, int input_channel, int output_channel, 
std::vector<HEaaN::Ciphertext> ctxt_bundle, 
std::vector<std::vector<std::vector<HEaaN::Plaintext>>> kernel_o) {
    int kernelsize;
    kernelsize = kernel_o[0][0].size();
    std::vector<HEaaN::Ciphertext> ctxt_out_bundle;
    // 비슷한 방법으로 input_channel은 ctxt_bundle에서 읽어올 수 있음.

    std::vector<std::vector<HEaaN::Ciphertext>> rotated_ctxts_bundle;
    for (int inputid = 0; inputid < (input_channel); ++inputid) {
        // Make rotated ctxts
        std::vector<HEaaN::Ciphertext> rotated_ctxts;
        HEaaN::Ciphertext rotated_ctxts_cache(context);
        eval.leftRotate(ctxt_bundle[inputid], -(imgsize + (gap * stride)), rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], -imgsize, rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], -(imgsize-(gap * stride)), rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], -(gap * stride), rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], 0, rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], (gap * stride), rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], imgsize-(gap * stride), rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], imgsize, rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);
        eval.leftRotate(ctxt_bundle[inputid], imgsize+(gap * stride), rotated_ctxts_cache);
        rotated_ctxts.push_back(rotated_ctxts_cache);

        rotated_ctxts_bundle.push_back(rotated_ctxts);
    }

    // Convolution
    for (int outputid = 0; outputid < output_channel; ++outputid) {
        // std::cout << outputid << " out\n";

        HEaaN::Ciphertext ctxt_out(context);
        for (int inputid = 0; inputid < (input_channel); ++inputid) {
            // std::cout << inputid << " in\n";
            HEaaN::Ciphertext ctxt_out_cache(context);
            if (kernelsize == 9) {
                HEaaN::Ciphertext mult_cache(context);
                eval.multWithoutRescale(rotated_ctxts_bundle[inputid][0], kernel_o[outputid][inputid][0], ctxt_out_cache);
                // ctxt_out_cache = mult_cache;
                for (int i = 1; i < 9; ++i) {
                    eval.multWithoutRescale(rotated_ctxts_bundle[inputid][i], kernel_o[outputid][inputid][i], mult_cache);
                    eval.add(ctxt_out_cache, mult_cache, ctxt_out_cache);
                }

            } else if (kernelsize == 1) {
                eval.multWithoutRescale(ctxt_bundle[inputid], kernel_o[outputid][inputid][0], ctxt_out_cache);
            
            } else {
                std::cout << "The size of Kernel bundle or Ctxt bundle is not proper!" << "\n";
                exit(1);
            }


            if (inputid == 0) {
                ctxt_out = ctxt_out_cache;
            } else {
                eval.add(ctxt_out, ctxt_out_cache, ctxt_out);
            }
        }
        eval.rescale(ctxt_out);
        ctxt_out_bundle.push_back(ctxt_out);
    }

    return ctxt_out_bundle;
}


