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


