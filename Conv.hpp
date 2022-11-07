////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


HEaaN::Ciphertext Conv(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int imgsize, int gap, int stride, HEaaN::Ciphertext ctxt, 
std::vector<HEaaN::Message> kernel_bundle) {
    int kernelsize;
    kernelsize = kernel_bundle.size();
    HEaaN::Ciphertext ctxt_out(context);
    
    if (kernelsize == 9) {
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
        eval.mult(rotated_ctxts[0], kernel_bundle[0], rotated_ctxts[0]);
        ctxt_out = rotated_ctxts[0];
        for (int i = 1; i < 9; ++i) {
            eval.mult(rotated_ctxts[i], kernel_bundle[i], rotated_ctxts[i]);
            eval.add(ctxt_out, rotated_ctxts[i], ctxt_out);
        }
    } else if (kernelsize == 1) {
        eval.mult(ctxt, kernel_bundle[0], ctxt_out);
    } else {
        std::cout << "The size of Kernel bundle is not 1 or 9!" << "\n";
        exit(1);
    }

    return ctxt_out;
}


