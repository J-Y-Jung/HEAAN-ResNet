#pragma once
#include "Conv.hpp"
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

namespace {
    using namespace HEaaN;
    using namespace std;
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                     //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed Withno the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


std::vector<HEaaN::Ciphertext> Conv(HEaaN::Context context, HEaaN::KeyPack pack,
    HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, int input_channel, int output_channel,
    std::vector<HEaaN::Ciphertext>& ctxt_bundle,
    std::vector<std::vector<std::vector<HEaaN::Plaintext>>>& kernel_o) {

    int kernelsize = kernel_o[0][0].size();

    HEaaN::Ciphertext ctxt_init(context);

    std::vector<HEaaN::Ciphertext> ctxt_out_bundle(output_channel, ctxt_init);


    // Convolution

    if (kernelsize == 9) {

        std::vector<std::vector<HEaaN::Ciphertext>> rotated_ctxts_bundle(input_channel, std::vector<HEaaN::Ciphertext>(9, ctxt_init));

        //#pragma omp parallel for collapse(3)
        for (int inputid = 0; inputid < (input_channel); ++inputid) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    eval.leftRotate(ctxt_bundle[inputid], (i - 1) * imgsize + (j - 1) * (gap * stride), rotated_ctxts_bundle[inputid][3 * i + j]);
                }
            }
        }

        //#pragma omp parallel for
        for (int outputid = 0; outputid < output_channel; ++outputid) {
            ctxt_out_bundle[outputid] = auxiliaryFtn9(eval, context, rotated_ctxts_bundle, kernel_o[outputid], input_channel);
        }

        rotated_ctxts_bundle.clear();
        rotated_ctxts_bundle.shrink_to_fit();

    }

    else {

        //#pragma omp parallel for
        for (int outputid = 0; outputid < output_channel; ++outputid) {
            ctxt_out_bundle[outputid] = auxiliaryFtn1(eval, context, ctxt_bundle, kernel_o[outputid], input_channel);
        }
    }

    return ctxt_out_bundle;

}
