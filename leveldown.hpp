#pragma once
#include <iostream>
#include <omp.h>
#include "HEaaN/heaan.hpp"

namespace {
//using namespace HEaaN::;
using namespace std;
}

void levelDownBundle(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
					vector<vector<HEaaN::Ciphertext>>& ctxt_bundle, u64 target_level){
	u64 d1 = ctxt_bundle.size();
	u64 d2 = ctxt_bundle[0].size();
	#pragma omp parallel for collapse(2)
	for (u64 i=0; i<d1; i++){
		for (u64 j=0; j<d2; j++){
			eval.levelDown(ctxt_bundle[i][j], target_level, ctxt_bundle[i][j]);
		}
	}
	return;
}

void levelDownVector(HEaaN::Context context, HEaaN::KeyPack pack, HEaaN::HomEvaluator eval, 
					vector<HEaaN::Ciphertext>& ctxt_vector, u64 target_level){
	u64 d1 = ctxt_bundle.size();
	#pragma omp parallel for
	for (u64 i=0; i<d1; i++){
		eval.levelDown(ctxt_bundle[i], target_level, ctxt_bundle[i]);
	}
	return;
}
