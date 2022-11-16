#pragma once
#include <iostream>
#include "HEaaN/heaan.hpp"
#include "rotsum.hpp"

namespace {
//using namespace HEaaN::;
using namespace std;
}

HEaaN::Ciphertext Avgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, HEaaN::Ciphertext ctxt) {
    ctxt = RotSumToIdx(context, pack, eval, 4*32, 3, 0, ctxt);
    ctxt = RotSumToIdx(context, pack, eval, 4, 3, 0, ctxt);
    return ctxt;
}

//we aggregate scaling in Avgpool by 1/64 to the linear weight vectors in preprocessing 
HEaaN::Ciphertext FC64(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
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