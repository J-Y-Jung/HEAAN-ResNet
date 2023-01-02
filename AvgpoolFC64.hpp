#pragma once
#include <iostream>
#include <omp.h>
#include "HEaaN/heaan.hpp"
#include "rotsum.hpp"

namespace {
//using namespace HEaaN::;
using namespace std;
}

HEaaN::Ciphertext singleAvgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, HEaaN::Ciphertext ctxt);
HEaaN::Ciphertext Avgpool(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, vector<HEaaN::Ciphertext> ctxt);
HEaaN::Ciphertext oldFC64(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              HEaaN::Ciphertext ctxt_1, HEaaN::Ciphertext ctxt_2, vector<HEaaN::Plaintext> ptxt_vec_1, vector<HEaaN::Plaintext> ptxt_vec_2);
HEaaN::Ciphertext oldFC64Packed(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              vector<HEaaN::Ciphertext> ctxt_1, vector<HEaaN::Ciphertext> ctxt_2, vector<HEaaN::Plaintext> ptxt_vec_1, vector<HEaaN::Plaintext> ptxt_vec_2);
vector<HEaaN::Ciphertext> FC64(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
                              vector<HEaaN::Ciphertext> ctxt, vector<vector<HEaaN::Plaintext>> ptxt, vector<HEaaN::Plaintext> bias);