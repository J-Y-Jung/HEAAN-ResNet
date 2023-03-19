#pragma once
#include <iostream>
#include <omp.h>
#include "HEaaN/heaan.hpp"
#include "rotsum.hpp"
#include "leveldown.hpp"

namespace {
using namespace HEaaN;
using namespace std;
}

HEaaN::Ciphertext singleAvgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, HEaaN::Ciphertext &ctxt) {
    ctxt = RotSumToIdx(context, pack, eval, 4*32, 3, 0, ctxt);
    ctxt = RotSumToIdx(context, pack, eval, 4, 3, 0, ctxt);
    return ctxt;
}

HEaaN::Ciphertext singleAvgpool2(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, HEaaN::Ciphertext &ctxt) {
    Ciphertext ctxt_temp(context);
    eval.leftRotate(ctxt, 4*32*4, ctxt_temp);
    eval.add(ctxt, ctxt_temp, ctxt);
    eval.leftRotate(ctxt, 4*32*2, ctxt_temp);
    eval.add(ctxt, ctxt_temp, ctxt);
    eval.leftRotate(ctxt, 4*32*1, ctxt_temp);
    eval.add(ctxt, ctxt_temp, ctxt);
    eval.leftRotate(ctxt, 4*4, ctxt_temp);
    eval.add(ctxt, ctxt_temp, ctxt);
    eval.leftRotate(ctxt, 4*2, ctxt_temp);
    eval.add(ctxt, ctxt_temp, ctxt);
    eval.leftRotate(ctxt, 4*1, ctxt_temp);
    eval.add(ctxt, ctxt_temp, ctxt);
    return ctxt;
}

vector<HEaaN::Ciphertext> Avgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, vector<HEaaN::Ciphertext> &ctxt) {
	levelDownVector(context,pack,eval,ctxt, 1);
    u64 n = ctxt.size();
    #pragma omp parallel for
    for (u64 i=0; i<n;i++){
        ctxt[i] = singleAvgpool2(context, pack, eval, ctxt[i]);
    }
	return ctxt;
}

vector<HEaaN::Ciphertext> FC64p(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              vector<HEaaN::Ciphertext> &ctxt, vector<vector<HEaaN::Plaintext>> &ptxt, vector<HEaaN::Plaintext> &bias) {
    HEaaN::Ciphertext zero_ct(context);
    vector<HEaaN::Ciphertext> v(64, zero_ct);
    vector<HEaaN::Ciphertext> out(10, zero_ct);
    vector<vector<HEaaN::Ciphertext>> tmp(10, v);
    int level1 = ctxt[0].getLevel();
    int level2 = ptxt[0][0].getLevel();
    for (u64 i=0;i<10 ;i++){
	#pragma omp parallel for 
        for (u64 j=0;j<64 ;j++){
		if(level1 != level2){
				ptxt[i][j].setLevel(level1);
			}
            eval.mult(ctxt[j], ptxt[i][j], tmp[i][j]);
        }
		for (u64 j=1;j<64 ;j++){
			eval.add(tmp[i][0], tmp[i][j], tmp[i][0]);
		}
	}
	#pragma omp parallel for 
	for (u64 i=0;i<10 ;i++){
        eval.add(tmp[i][0], bias[i], out[i]);
    }
    
    return out;
}
