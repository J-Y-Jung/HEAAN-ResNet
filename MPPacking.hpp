////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


//HEaaN::Ciphertext MPPacking(HEaaN::Context context, HEaaN::KeyPack pack,
//HEaaN::HomEvaluator eval, int imgsize, 
//std::vector<HEaaN::Ciphertext>& ctxt_bundle) {
    //int num_ctxt;
    //num_ctxt = ctxt_bundle.size();
    //if (floor(sqrt(num_ctxt)) != (double)sqrt(num_ctxt)) {
    //    std::cout << "Ciphertext bundle size is NOT square!" << "\n";
    //    exit(1);
    //}
    //int gap = (int)sqrt(num_ctxt);
    // std::cout << gap << "\n";
	
	// Rotate ctxts
    
    // HEaaN::Ciphertext ctxt_rotated_bundle_cache(context);
	//HEaaN::Ciphertext ctxt_init(context);
	//std::vector<HEaaN::Ciphertext> ctxt_rotated_bundle(4, ctxt_init);
		
	//#pragma omp parallel for collapse(2)
	//for (int i = 0; i < 2; ++i) {
        //for (int j = 0; j < 2; ++j) {
			//eval.leftRotate(ctxt_bundle[2*i+j], -(imgsize*i)-j, ctxt_rotated_bundle[2*i+j]);
		//}
	//}
	
	//ctxt_bundle.clear();
	//ctxt_bundle.shrink_to_fit();
	
    // Sum
    //HEaaN::Ciphertext ctxt_sum(context);
    //	if (ctxt_bundle.size() != 4){
//		std::cout << "Ciphertext bundle size is NOT 4!" << "\n";
//	}
//	
//	#pragma omp parallel
//	{
//		eval.add(ctxt_rotated_bundle[0], ctxt_rotated_bundle[1], ctxt_rotated_bundle[0]);
//		eval.add(ctxt_rotated_bundle[2], ctxt_rotated_bundle[3], ctxt_rotated_bundle[2]);
//	}
//	eval.add(ctxt_rotated_bundle[0], ctxt_rotated_bundle[2], ctxt_sum);
//
//
//	ctxt_rotated_bundle.clear();
//	ctxt_rotated_bundle.shrink_to_fit();
//
//  return ctxt_sum;
//}

HEaaN::Ciphertext MPPacking1(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int imgsize, 
std::vector<HEaaN::Ciphertext>& ctxt_bundle) {
	if (ctxt_bundle.size() != 4){
		std::cout << "Ciphertext bundle size is NOT 4!" << "\n";
	}
	
	// Rotate ctxts
    
    // HEaaN::Ciphertext ctxt_rotated_bundle_cache(context);
	HEaaN::Ciphertext ctxt_init(context);
	std::vector<HEaaN::Ciphertext> ctxt_rotated_bundle(4, ctxt_init);
		
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < 2; ++i) {
        	for (int j = 0; j < 2; ++j) {
			eval.leftRotate(ctxt_bundle[2*i+j], -(imgsize*i)-j, ctxt_rotated_bundle[2*i+j]);
		}
	}
	
	ctxt_bundle.clear();
	ctxt_bundle.shrink_to_fit();
	
    // Sum
    HEaaN::Ciphertext ctxt_sum(context);
   
	#pragma omp parallel
	{
		eval.add(ctxt_rotated_bundle[0], ctxt_rotated_bundle[1], ctxt_rotated_bundle[0]);
		eval.add(ctxt_rotated_bundle[2], ctxt_rotated_bundle[3], ctxt_rotated_bundle[2]);
	}
	eval.add(ctxt_rotated_bundle[0], ctxt_rotated_bundle[2], ctxt_sum);
	

	ctxt_rotated_bundle.clear();
	ctxt_rotated_bundle.shrink_to_fit();

    return ctxt_sum;
}

HEaaN::Ciphertext MPPacking2(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int imgsize, 
std::vector<HEaaN::Ciphertext>& ctxt_bundle) {
	
	// Rotate ctxts
    
    // HEaaN::Ciphertext ctxt_rotated_bundle_cache(context);
	HEaaN::Ciphertext ctxt_init(context);
	std::vector<HEaaN::Ciphertext> ctxt_rotated_bundle(4, ctxt_init);
		
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < 2; ++i) {
        	for (int j = 0; j < 2; ++j) {
			eval.leftRotate(ctxt_bundle[2*i+j], -(imgsize*2*i)-j*2, ctxt_rotated_bundle[2*i+j]);
		}
	}
	
	ctxt_bundle.clear();
	ctxt_bundle.shrink_to_fit();
	
    // Sum
    HEaaN::Ciphertext ctxt_sum(context);
   
	#pragma omp parallel
	{
		eval.add(ctxt_rotated_bundle[0], ctxt_rotated_bundle[1], ctxt_rotated_bundle[0]);
		eval.add(ctxt_rotated_bundle[2], ctxt_rotated_bundle[3], ctxt_rotated_bundle[2]);
	}
	eval.add(ctxt_rotated_bundle[0], ctxt_rotated_bundle[2], ctxt_sum);
	

	ctxt_rotated_bundle.clear();
	ctxt_rotated_bundle.shrink_to_fit();

    return ctxt_sum;
}
