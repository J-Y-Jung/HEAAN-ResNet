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
	#pragma omp for collapse(2)
	for (u64 i; i<d1; i++){
		for (u64 j; j<d2; j++){
			eval.levelDown(ctxt_bundle[i][j], target_level, ctxt_bundle[i][j]);
		}
	}
	return;
}