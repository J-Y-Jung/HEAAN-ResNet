////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


HEaaN::Ciphertext MPPacking(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int imgsize, int gap, 
HEaaN::Ciphertext ctxt0, HEaaN::Ciphertext ctxt1, HEaaN::Ciphertext ctxt2, HEaaN::Ciphertext ctxt3) {
    
    // Save as ctxt bundle
    std::vector<HEaaN::Ciphertext> ctxt_bundle;
    ctxt_bundle.push_back(ctxt0);
    ctxt_bundle.push_back(ctxt1);
    ctxt_bundle.push_back(ctxt2);
    ctxt_bundle.push_back(ctxt3);
    
    
    const auto log_slots = getLogFullSlots(context);
    
    // Making MPP masks
    std::vector<HEaaN::Message> mask_bundle;
    HEaaN::Message mask(log_slots);
    std::optional<size_t> num;
    size_t length = num.has_value() ? num.value() : mask.getSize();

    for (size_t idx = 0; idx < length; ++idx) {
        mask[idx].real(0.0);
        mask[idx].imag(0.0);
    }
    for (size_t idx = 0; idx < length; ++idx) {
        mask[idx].real(1.0);
        mask[idx].imag(0.0);
        idx = idx + gap - 1;
    }
    // printMessage(mask);
    for (int i = 0; i < length; ++i) {
        if ((i/imgsize) % 2 == 1) {
            mask[i].real(0.0);
            mask[i].imag(0.0);
            // std::cout << i << '\n';
        }
    }
    printMessage(mask);
    // mask_bundle.push_back(mask);



    // Masking
    std::vector<HEaaN::Ciphertext> ctxt_masked_bundle;
    HEaaN::Ciphertext ctxt_masked_cache(context);
    for (int i = 0; i < (gap*gap); i++) {
        eval.mult(ctxt_bundle[i], mask, ctxt_masked_cache);
        ctxt_masked_bundle.push_back(ctxt_masked_cache);
    }

    // Rotate masked ctxts
    std::vector<HEaaN::Ciphertext> ctxt_rotated_bundle;
    HEaaN::Ciphertext ctxt_rotated_bundle_cache(context);

    ctxt_rotated_bundle.push_back(ctxt_masked_bundle[0]);
    eval.leftRotate(ctxt_masked_bundle[1], -1, ctxt_rotated_bundle_cache);
    ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
    eval.leftRotate(ctxt_masked_bundle[2], -imgsize, ctxt_rotated_bundle_cache);
    ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
    eval.leftRotate(ctxt_masked_bundle[3], -imgsize-1, ctxt_rotated_bundle_cache);
    ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);

    // Sum
    HEaaN::Ciphertext ctxt_sum(context);
    ctxt_sum = ctxt_rotated_bundle[0];
    for (int i = 1; i < (gap*gap); i++) {
        eval.add(ctxt_sum, ctxt_rotated_bundle[i], ctxt_sum);
    }



    return ctxt_sum;
}
