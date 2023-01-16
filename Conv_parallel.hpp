#pragma once
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                     //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed Withno the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


std::vector<HEaaN::Ciphertext> Conv_parallel(HEaaN::Context context, HEaaN::KeyPack pack,
    HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, int input_channel, int output_channel,
    std::vector<HEaaN::Ciphertext>& ctxt_bundle,
    std::vector<std::vector<std::vector<HEaaN::Plaintext>>>& kernel_o) {
    int kernelsize;
    kernelsize = kernel_o[0][0].size();

    HEaaN::Ciphertext ctxt_init(context);

    std::vector<HEaaN::Ciphertext> ctxt_out_bundle;
    // 비슷한 방법으로 input_channel은 ctxt_bundle에서 읽어올 수 있음.

    std::vector<std::vector<HEaaN::Ciphertext>> rotated_ctxts_bundle(input_channel, std::vector<HEaaN::Ciphertext>(9, ctxt_init));

    #pragma omp parallel for collapse(3)
    for (int inputid = 0; inputid < (input_channel); ++inputid) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                eval.leftRotate(ctxt_bundle[inputid], (i-1)*imgsize + (j-1) * (gap * stride), rotated_ctxts_bundle[inputid][3*i+j]);
            }
        }
    }

    // Convolution

    // Convolution
    for (int outputid = 0; outputid < output_channel; ++outputid) {
        // std::cout << outputid << " out\n";

        HEaaN::Ciphertext ctxt_out(context);
        for (int inputid = 0; inputid < (input_channel); ++inputid) {
            //std::cout << inputid << " in\n";
            HEaaN::Ciphertext ctxt_out_cache(context);
            if (kernelsize == 9) {
                HEaaN::Ciphertext mult_cache(context);
                int level1 = rotated_ctxts_bundle[inputid][0].getLevel();
                int level2 = kernel_o[outputid][inputid][0].getLevel();
                if (level1 > level2) {
                    eval.levelDown(rotated_ctxts_bundle[inputid][0], level2, rotated_ctxts_bundle[inputid][0]);
                }
                else if (level1 < level2) {
                    kernel_o[outputid][inputid][0].setLevel(level1);
                }
                //eval.levelDown(rotated_ctxts_bundle[inputid][0],kernel_o[outputid][inputid][0].getLevel(),rotated_ctxts_bundle[inputid][0]); //추가
                eval.multWithoutRescale(rotated_ctxts_bundle[inputid][0], kernel_o[outputid][inputid][0], ctxt_out_cache);
                // ctxt_out_cache = mult_cache;
                for (int i = 1; i < 9; ++i) {
                    int level1 = rotated_ctxts_bundle[inputid][i].getLevel();
                    int level2 = kernel_o[outputid][inputid][i].getLevel();
                    if (level1 > level2) {
                        eval.levelDown(rotated_ctxts_bundle[inputid][i], level2, rotated_ctxts_bundle[inputid][i]);
                    }
                    else if (level1 < level2) {
                        kernel_o[outputid][inputid][i].setLevel(level1);
                    }
                    eval.multWithoutRescale(rotated_ctxts_bundle[inputid][i], kernel_o[outputid][inputid][i], mult_cache);
                    eval.add(ctxt_out_cache, mult_cache, ctxt_out_cache);
                }

            }
            else if (kernelsize == 1) {
                int level1 = ctxt_bundle[inputid].getLevel();
                int level2 = kernel_o[outputid][inputid][0].getLevel();
                if (level1 > level2) {
                    eval.levelDown(ctxt_bundle[inputid], level2, ctxt_bundle[inputid]);
                }
                else if (level1 < level2) {
                    kernel_o[outputid][inputid][0].setLevel(level1);
                }
                eval.multWithoutRescale(ctxt_bundle[inputid], kernel_o[outputid][inputid][0], ctxt_out_cache);

            }
            else {
                std::cout << "The size of Kernel bundle or Ctxt bundle is not proper!" << "\n";
                exit(1);
            }


            if (inputid == 0) {
                ctxt_out = ctxt_out_cache;
            }
            else {
                eval.add(ctxt_out, ctxt_out_cache, ctxt_out);
            }
        }
        eval.rescale(ctxt_out);
        ctxt_out_bundle.push_back(ctxt_out);
    }
    rotated_ctxts_bundle.clear();
    rotated_ctxts_bundle.shrink_to_fit();

    return ctxt_out_bundle;
}


