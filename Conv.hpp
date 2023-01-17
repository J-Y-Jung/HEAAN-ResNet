namespace {
    using namespace std;
    using namespace HEaaN;
}


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
std::vector<HEaaN::Ciphertext>& ctxt_bundle, 
std::vector<std::vector<std::vector<HEaaN::Plaintext>>>& kernel_o) {

    int kernelsize;
    kernelsize = kernel_o[0][0].size();

    HEaaN::Ciphertext ctxt_init(context);

    std::vector<HEaaN::Ciphertext> ctxt_out_bundle;
    // 비슷한 방법으로 input_channel은 ctxt_bundle에서 읽어올 수 있음.


    if (kernelsize == 9) {

        std::vector<std::vector<HEaaN::Ciphertext>> rotated_ctxts_bundle(input_channel, std::vector<HEaaN::Ciphertext>(9, ctxt_init));

        #pragma omp parallel for collapse(3)
        for (int inputid = 0; inputid < (input_channel); ++inputid) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    eval.leftRotate(ctxt_bundle[inputid], (i - 1) * imgsize + (j - 1) * (gap * stride), rotated_ctxts_bundle[inputid][3 * i + j]);
                }
            }
        }

        for (int outputid = 0; outputid < output_channel; ++outputid) {

            HEaaN::Ciphertext ctxt_out(context);

            for (int inputid = 0; inputid < (input_channel); ++inputid) {

                HEaaN::Ciphertext ctxt_out_cache(context);

                eval.multWithoutRescale(rotated_ctxts_bundle[inputid][0], kernel_o[outputid][inputid][0], ctxt_out_cache);

                HEaaN::Ciphertext mult_cache(context);
                for (int i = 1; i < 9; ++i) {
                    eval.multWithoutRescale(rotated_ctxts_bundle[inputid][i], kernel_o[outputid][inputid][i], mult_cache);
                    eval.add(ctxt_out_cache, mult_cache, ctxt_out_cache);
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

    }
    

    if (kernelsize == 1) {

        for (int outputid = 0; outputid < output_channel; ++outputid) {
            // std::cout << outputid << " out\n";

            HEaaN::Ciphertext ctxt_out(context);

            for (int inputid = 0; inputid < (input_channel); ++inputid) {

                HEaaN::Ciphertext ctxt_out_cache(context);
                eval.multWithoutRescale(ctxt_bundle[inputid], kernel_o[outputid][inputid][0], ctxt_out_cache);

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


    }


    return ctxt_out_bundle;
}


