#pragma once
#include <iostream>
#include "HEaaN/heaan.hpp"
#include "rotsum.hpp"

namespace {
//using namespace HEaaN::;
using namespace std;
}
//old version and new version (which packs 16 images in each ctxt) shares same Avgpool.
HEaaN::Ciphertext Avgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, HEaaN::Ciphertext ctxt) {
    ctxt = RotSumToIdx(context, pack, eval, 4*32, 3, 0, ctxt);
    ctxt = RotSumToIdx(context, pack, eval, 4, 3, 0, ctxt);
    return ctxt;
}

//we aggregate scaling in Avgpool by 1/64 to the linear weight vectors in preprocessing 
//old version assuming 16-channel multiplexed packing
HEaaN::Ciphertext FC64old(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              HEaaN::Ciphertext ctxt, vector<HEaaN::Plaintext> ptxt_vec) {
    HEaaN::Ciphertext ctxt_tmp(context), ctxt_out(context);
    eval.mult(ctxt, ptxt_vec[0], ctxt_out);
    for (u64 i=1;i<10 ;i++){
        eval.mult(ctxt, ptxt_vec[i], ctxt_tmp);
        eval.leftRotate(ctxt_tmp, -(8*(i%4)+8*32*(i>>2)),ctxt_tmp);
        eval.add(ctxt_out, ctxt_tmp, ctxt_out);
        //correct values are stored in indices (0, 8, 16, 24, 256, 264, 272, 280, 512, 520)
    } 
    ctxt_out = RotSumToIdx(context, pack, eval, 1, 2, 0, ctxt_out);
    ctxt_out = RotSumToIdx(context, pack, eval, 32, 2, 0, ctxt_out);
    ctxt_out = RotSumToIdx(context, pack, eval, 32*32, 2, 0, ctxt_out);
    return ctxt_out;
}
//new version assuming 16 images in one 32*32 block, and 32 channels in one ciphertext.

HEaaN::Ciphertext FC64(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              HEaaN::Ciphertext ctxt_1, HEaaN::Ciphertext ctxt_2, vector<HEaaN::Plaintext> ptxt_vec_1, vector<HEaaN::Plaintext> ptxt_vec_2) {
    //ctxt_1 and ptxt_vec_1 accounts for channel 1 ~ 32, ctxt_2 and ptxt_vec_2 accounts for channel 33 ~ 64.
    HEaaN::Ciphertext ctxt_tmp(context), ctxt_out(context);
    eval.mult(ctxt_1, ptxt_vec_1[0], ctxt_out);
    for (u64 i=1;i<10 ;i++){
        eval.mult(ctxt_1, ptxt_vec_1[i], ctxt_tmp);
        eval.leftRotate(ctxt_tmp, -(4*(i%8)+4*32*(i>>3)),ctxt_tmp);
        eval.add(ctxt_out, ctxt_tmp, ctxt_out);
        //correct values are stored in indices (0, 4, 8, 12, 16, 20, 24, 28, 128, 132)
    } 
    for (u64 i=0;i<10 ;i++){
        eval.mult(ctxt_2, ptxt_vec_2[i], ctxt_tmp);
        eval.leftRotate(ctxt_tmp, -(4*(i%8)+4*32*(i>>3)),ctxt_tmp);
        eval.add(ctxt_out, ctxt_tmp, ctxt_out);
        //correct values (for the first image) are stored in indices (0, 4, 8, 12, 16, 20, 24, 28, 128, 132)
    } 
    ctxt_out = RotSumToIdx(context, pack, eval, 32*32, 5, 0, ctxt_out);
    return ctxt_out;
}