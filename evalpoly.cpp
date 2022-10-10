
#pragma once
#include <iostream>
#include "HEaaN/heaan.hpp"
#include <cmath>

void SetUp(HEaaN::HomEvaluator eval, HEaaN::Context context, HEaaN::Ciphertext ctxt,
           std::vector<HEaaN::Ciphertext>& BS_basis,
           std::vector<HEaaN::Ciphertext>& GS_basis,
           const int length, const int m) {
    vec[0] = 1;
    vec[1] = ctxt;
    for (int i = 2; i < length; ++i) {
        if (i % 2 == 0) {
            HEaaN::Ciphertext ctxt_out(context);
            eval.mult(BS_basis[i / 2], BS_basis[i / 2], ctxt_out);
            BS_basis[i] = ctxt_out;
        }
        else {
            HEaaN::Ciphertext ctxt_out(context);
            eval.mult(BS_basis[(i - 1) / 2], BS_basis[(i + 1) / 2], ctxt_out);
            BS_basis[i] = ctxt_out;
        }
    }
    for (int j = 0; j < m - log2(length); ++j) {
        if (j == 0) {
            HEaaN::Ciphertext ctxt_out(context);
            eval.mult(BS_basis[length - 1], BS_basis[length - 1], ctxt_out);
            GS_basis[j] = ctxt_out;
        }
        else {
            HEaaN::Ciphertext ctxt_out(context);
            eval.mult(GS_basis[j], GS_basis[j], ctxt_out);
            GS_basis[j] = ctxt_out;
        }
    }
}

//BabyStep algo in Han-Ki.
HEaaN::Ciphertext BabyStep(HEaaN::Context context, HEaaN::HomEvaluator eval,
                           std::vector<double> polynomial,
                           std::vector<HEaaN::Ciphertext> & vec,
                           const int length) {

    HEaaN::Ciphertext ctxt_out(context);

    for (int i = 0; i < length; ++i) {
        HEaaN::Ciphertext ctxt_temp(context);
        eval.mul(polynomial[i], vec[i], ctxt_temp);
        eval.add(ctxt_out, ctxt_temp, ctxt_out);
    }
    return ctxt_out;
}


std::vector<double> vecSlice(vector<double> input, int a, int b) {
    return vector<double>(input.begin() + a, inp.begin() + b);
}

HEaaN::Ciphertext GiantStep(HEaaN::Context context, HEaaN::HomEvaluator eval, 
                            std::vector<double> &polynomial, 
                            const std::vector<HEaaN::Ciphertext> &BS_basis, 
                            const std::vector<HEaaN::Ciphertext> &GS_basis, 
                            int length, int m) {

    HEaaN::Ciphertext ctxt_result(context);

    if (polynomial.size() < 2**length) {
        ctxt_result = BabyStep(context, eval, polynomial, BS_basis, l);
        return ctxt_result
    }

    int k = floor(log2(polynomial.size()));

    std::vector<double> quotient = vecSlice(2**k, polynomial.size());
    std::vector<double> remainder = vecSlice(p, 0, 2**k);

    HEaaN::Ciphertext ctxt_quotient(context);
    HEaaN::Ciphertext ctxt_remainder(context);

    ctxt_quotient = GiantStep(context, eval, quotient, BS_basis, GS_basis, int l, int m);
    ctxt_remainder = GiantStep(context, eval, remainder, BS_basis, GS_basis, int l, int m);

    HEaaN::Ciphertext ctxt_temp(context);
    eval.mult(ctxt_quotient, GS_basis[k - l], ctxt_temp);
    eval.add(ctxt_temp, ctxt_remainder, ctxt_result);

    return ctxt_result
}


HEaaN::Ciphertext evalPolynomial(HEaaN::Ciphertext ctxt, std::vector<double> &polynomial) {

    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    std::cout << "Parameter : " << presetNamer(preset) << std::endl
        << std::endl;

    const auto log_slots = getLogFullSlots(context);

    // Generate a new secret key
    HEaaN::SecretKey sk(context);

    /*
    You can also use the constuctors
    SecretKey(const Context &context, std::istream &stream) or
    SecretKey(const Context &context, const std::string &key_dir_path)
    if you have the saved secret key file.
    */

    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... ";
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl;

    std::cout << "Generate multiplication key ... ";
    keygen.genMultiplicationKey();
    std::cout << "done" << std::endl;

    std::cout << "Generate rotation key ... ";
    keygen.genRotationKeyBundle();
    std::cout << "done" << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);
    HEaaN::HomEvaluator eval(context, pack);

    int l = ceil(log2(polynomial.size() + 1));
    int m = m / 2;

    std::vector<HEaaN::Ciphertext>& BS_basis = [];
    std::vector<HEaaN::Ciphertext>& GS_basis = [];

    Setup(context, eval, BS_basis, GS_basis, l, m);
    
    HEaaN::Ciphertext ctxt_out(context);
    
    ctxt_out = GiantStep(context, eval, polynomial, BS_basis, GS_basis, l, m);

}

HEaaN::Ciphertext AReLU(HEaaN::Ciphertext) {
    const std::vector<double>& polynomial_1 = {
    -3.38572283433492e-47, 2.49052143193754e01, 7.67064296707865e-45,
    -6.82383057582430e02, -1.33318527258859e-43, 6.80942845390599e03,
    9.19464568002043e-43, -3.12507100017105e04, -3.02547883089949e-42,
    7.47659388363757e04, 5.02426027571770e-42, -9.65076838475839e04,
    -4.05931240321443e-42, 6.36977923778246e04, 1.26671427827897e-42,
    -1.68602621347190e04 };

    const std::vector<double> &polynomial_2 = {
    -9.27991756967991e-46, 1.68285511926011e01  , 8.32408114686671e-44,
    -3.3981175049s5659e02 ,-1.27756566628511e-42 , 2.79069998793847e03 ,
    7.70152836729131e-42,-1.13514151573790e04  ,-2.41159918805990e-41,
    2.66230010283745e04 , 4.48807056213874e-41 ,-3.93840628661975e04 ,
    -5.34821622972202e-41, 3.87884230348060e04  , 4.25722502798559e-41,
    -2.62395303844988e04 ,-2.31146624263347e-41 , 1.23656207016532e04 ,
    8.58571463533718e-42,-4.05336460089999e03  ,-2.14564940301255e-42,
    9.06042880951087e02 , 3.44803367899992e-43 ,-1.31687649208388e02 ,
    -3.21717059336602e-44, 1.12176079033623e01  , 1.32425600403445e-45,
    -4.24938020467471e-01 };

    const std::vector<double> &polynomial_3 = { 
     6.72874968716530e-48, 5.31755497689391     , 5.68199275801086e-46,
    -3.54371531531577e01 ,-1.35187813155454e-44 , 1.84122441329140e02 ,
     1.05531766289589e-43,-6.55386830146253e02  ,-4.14266518871760e-43,
     1.63878335428060e03 , 9.63097361166316e-43 ,-2.95386237048226e03 ,
    -1.44556688409360e-42, 3.90806423362418e03  , 1.47265013864485e-42,
    -3.83496739165131e03 ,-1.04728251169615e-42 , 2.79960654766517e03 ,
     5.26108728786276e-43,-1.51286231886692e03  ,-1.86083902222546e-43,
     5.96160139340009e02 , 4.53644110199468e-44 ,-1.66321739302958e02 ,
    -7.25782287655313e-45, 3.10988369739884e01  , 6.85800520634485e-46,
    -3.49349374506190    ,-2.89849811206637e-47 , 1.78142156956495e-01
    };

    HEaaN::Ciphertext ctxt_out1(context);
    ctxt_out1 = evalPolynomial(ctxt, polynomial_1);

    HEaaN::Ciphertext ctxt_out2(context);
    ctxt_out2 = evalPolynomial(ctxt_out1, polynomial_2);

    // may need bootstrapping once

    HEaaN::Ciphertext ctxt_out(context);
    ctxt_out = evalPolynomial(ctxt_out2, polynomial_3);

    return ctxt_out;
}
