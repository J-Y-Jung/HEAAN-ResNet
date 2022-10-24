#pragma once
#include <iostream>
#include "HEaaN/heaan.hpp"

namespace {
using namespace HEaaN;
using namespace std;
}
HEaaN::Ciphertext RotSumToIdx(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              u64 rot_interval, u64 log_rot_num, u64 idx, HEaaN::Ciphertext ctxt) {
    //Rotate-sum 2^(log_rot_num)*rotated ciphertexts, which are rotated at the interval of rot_interval, 
    //and save the result starting at idx-th slot.
    //summing all elements of ciphertext is equivalent to
    //auto slots = ctxt.getSize();
    //auto log_slots = getLogFullSlots(context);
    HEaaN::Ciphertext ctxt_temp(context);
    //HEaaN::Ciphertext rotated_ctxts_cache(context) 
    if (idx % rot_interval != 0){
        for (int i=log_rot_num-1 ; i>=0; i--){
            eval.leftRotate(ctxt, (1<<i)*rot_interval, ctxt_temp);
            eval.add(ctxt, ctxt_temp, ctxt);
        }
        eval.leftRotate(ctxt, -idx, ctxt);
    }
    else if (idx % rot_interval == 0){
        idx = idx/rot_interval;
        for (int i=log_rot_num-1 ; i>=0; i--){
            eval.leftRotate(ctxt, (1<<i)*rot_interval*(-2*((idx>>i)%2)+1), ctxt_temp);
            eval.add(ctxt, ctxt_temp, ctxt);
        }
    }
    return ctxt;
}
