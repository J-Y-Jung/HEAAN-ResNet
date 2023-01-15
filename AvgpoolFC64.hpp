#pragma once
#include <iostream>
#include <omp.h>
#include "HEaaN/heaan.hpp"
#include "rotsum.hpp"
#include "leveldown.hpp"

namespace {
//using namespace HEaaN::;
using namespace std;
}

HEaaN::Ciphertext singleAvgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, HEaaN::Ciphertext &ctxt) {
    ctxt = RotSumToIdx(context, pack, eval, 4*32, 3, 0, ctxt);
    ctxt = RotSumToIdx(context, pack, eval, 4, 3, 0, ctxt);
    return ctxt;
}

vector<HEaaN::Ciphertext> Avgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, vector<HEaaN::Ciphertext> &ctxt) {
	levelDownVector(context,pack,eval,ctxt, 1);
    #pragma omp parallel for
    for (u64 i=0; i<ctxt.size();i++){
        ctxt[i] = singleAvgpool(context, pack, eval, ctxt[i]);
    }
	return ctxt;
}


//we delete scaling in Avgpool by 1/64 and add multiplying 64 to bias vectors in preprocessing 
//old version assuming 16-channel multiplexed packing
/*
HEaaN::Ciphertext oldFC64old(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
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
}*/
//new version assuming 16 images in one 32*32 block, and 32 channels in one ciphertext.
/*
HEaaN::Ciphertext FC64Old(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
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
*/
/*
//takes only vector of avgpool2idx - processed ciphertexts
HEaaN::Ciphertext FC64PackedOld(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              vector<HEaaN::Ciphertext> ctxt_1, vector<HEaaN::Ciphertext> ctxt_2, vector<HEaaN::Plaintext> ptxt_vec_1, vector<HEaaN::Plaintext> ptxt_vec_2) {
    //ctxt_1 and ptxt_vec_1 accounts for channel 1 ~ 32, ctxt_2 and ptxt_vec_2 accounts for channel 33 ~ 64.
    HEaaN::Ciphertext ctxt_tmp(context), ctxt_out(context);
    eval.mult(ctxt_1[0], ptxt_vec_1[0], ctxt_out);
    for (u64 j=1;j<10 ;i++){
    for (u64 i=1;i<10 ;i++){
        eval.mult(ctxt_1[0], ptxt_vec_1[i], ctxt_tmp);
        eval.leftRotate(ctxt_tmp, -(4*(i%8)+4*32*(i>>3)),ctxt_tmp);
        eval.add(ctxt_out, ctxt_tmp, ctxt_out);
        //correct values are stored in indices (0, 4, 8, 12, 16, 20, 24, 28, 128, 132)
    } 
    for (u64 i=0;i<10 ;i++){
        eval.mult(ctxt_2[0], ptxt_vec_2[i], ctxt_tmp);
        eval.leftRotate(ctxt_tmp, -(4*(i%8)+4*32*(i>>3)),ctxt_tmp);
        eval.add(ctxt_out, ctxt_tmp, ctxt_out);
        //correct values (for the first image) are stored in indices (0, 4, 8, 12, 16, 20, 24, 28, 128, 132)
    } 
    
    ctxt_out = RotSumToIdx(context, pack, eval, 32*32, 5, 0, ctxt_out);
    return ctxt_out;
}
*/
//ctxt: vector of 64 ciphertext which has 16*32 images each and representing single channel
//ptxt:  vector of 10 vectors, and each sub-vector contains a 64 encoded values/constant
//with lazy rescaling
vector<HEaaN::Ciphertext> FC64(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
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
            eval.multWithoutRescale(ctxt[j], ptxt[i][j], tmp[i][j]);
        }
		for (u64 j=1;j<64 ;j++){
			eval.add(tmp[i][0], tmp[i][j], tmp[i][0]);
		}
	}
	#pragma omp parallel for 
	for (u64 i=0;i<10 ;i++){
        eval.rescale(tmp[i][0]);
        eval.add(tmp[i][0], bias[i], out[i]);
    }
    
    return out;
}
