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
HEaaN::HomEvaluator eval, int imgsize, 
std::vector<HEaaN::Ciphertext>& ctxt_bundle) {
    int num_ctxt;
    num_ctxt = ctxt_bundle.size();
    if (floor(sqrt(num_ctxt)) != (double)sqrt(num_ctxt)) {
        std::cout << "Ciphertext bundle size is NOT square!" << "\n";
        exit(1);
    }
    int gap = (int)sqrt(num_ctxt);
    // std::cout << gap << "\n";

    const auto log_slots = getLogFullSlots(context);
    
    // Making MPP masks
    std::vector<HEaaN::Message> mask_bundle;
    HEaaN::Message mask(log_slots);
    std::optional<size_t> num;
    // size_t length = num.has_value() ? num.value() : mask.getSize();

    for (size_t idx = 0; idx < mask.getSize(); ++idx) {
        mask[idx].real(0.0);
        mask[idx].imag(0.0);
    }
    for (size_t idx = 0; idx < mask.getSize(); ++idx) {
        mask[idx].real(1.0);
        mask[idx].imag(0.0);
        idx = idx + gap - 1;
    }
    // printMessage(mask);
    for (int i = 0; i < mask.getSize(); ++i) {
        if ((i/imgsize) % gap != 0) {
            mask[i].real(0.0);
            mask[i].imag(0.0);
            // std::cout << i << '\n';
        }
    }
    // printMessage(mask);
    // std::cout << mask[0] << mask[imgsize] << mask[2*imgsize] << mask[3*imgsize] << "\n";
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
    if (gap == 2) {
        ctxt_rotated_bundle.push_back(ctxt_masked_bundle[0]);
        eval.leftRotate(ctxt_masked_bundle[1], -1, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);

        eval.leftRotate(ctxt_masked_bundle[2], -imgsize, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[3], -imgsize-1, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
    } else if (gap == 4) {
        ctxt_rotated_bundle.push_back(ctxt_masked_bundle[0]);
        eval.leftRotate(ctxt_masked_bundle[1], -1, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[2], -2, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[3], -3, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);

        eval.leftRotate(ctxt_masked_bundle[4], -imgsize, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[5], -imgsize-1, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[6], -imgsize-2, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[7], -imgsize-3, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);

        eval.leftRotate(ctxt_masked_bundle[8], -(2*imgsize), ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[9], -(2*imgsize)-1, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[10], -(2*imgsize)-2, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[11], -(2*imgsize)-3, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);

        eval.leftRotate(ctxt_masked_bundle[12], -(3*imgsize), ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[13], -(3*imgsize)-1, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[14], -(3*imgsize)-2, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
        eval.leftRotate(ctxt_masked_bundle[15], -(3*imgsize)-3, ctxt_rotated_bundle_cache);
        ctxt_rotated_bundle.push_back(ctxt_rotated_bundle_cache);
    }
    ctxt_masked_bundle.clear();
    ctxt_masked_bundle.shrink_to_fit();

    // Sum
    HEaaN::Ciphertext ctxt_sum(context);
    ctxt_sum = ctxt_rotated_bundle[0];
    for (int i = 1; i < (gap*gap); i++) {
        eval.add(ctxt_sum, ctxt_rotated_bundle[i], ctxt_sum);
    }



    return ctxt_sum;
}
