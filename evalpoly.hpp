#pragma once
#include <iostream>
#include <cmath>
#include "HEaaN/heaan.hpp"

// Construct BabyStep basis and GiantStep basis.
void SetUp(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext ctxt,
           std::vector<HEaaN::Ciphertext> &BS_basis,
           std::vector<HEaaN::Ciphertext> &GS_basis,
           const int k, const int l) {

    //this is garbage value
    BS_basis.push_back(ctxt);
    
    //BS_basis consists of x, x, x^2, ... ,x^{k-1}; first entry is useless
    BS_basis.push_back(ctxt);
    
    HEaaN::Ciphertext ctxt_temp(context);

    for (int i = 2; i < k; ++i) {
        if (i% 2 == 0) {
            eval.mult(BS_basis[i/2], BS_basis[i/2], ctxt_temp);
            BS_basis.push_back(ctxt_out);
        }
        else {
            eval.mult(BS_basis[(i-1)/2], BS_basis[(i+1)/2], ctxt_temp);
            BS_basis.push_back(ctxt_out);
        }
    }

    HEaaN::Ciphertext ctxt_gs_init(context);
    eval.mult(BS_basis[k - 1], BS_basis[k - 1], ctxt_gs_init);
    GS_basis.push_back(ctxt_gs_init);

    for (int j = 1; j < l; ++j) {
        eval.mult(GS_basis[j-1], GS_basis[j-1], ctxt_temp);
        GS_basis.push_back(ctxt_out);
    }
}

//BabyStep algo in Han-Ki.
HEaaN::Ciphertext BabyStep(HEaaN::Context context, HEaaN::HomEvaluator eval,
                           const std::vector<HEaaN::Ciphertext> &basis,
                           const std::vector<double> &polynomial,
                           const int length) {

    HEaaN::Ciphertext ctxt_out(context);
    eval.mult(basis[length-1], polynomial[length-1], ctxt_out);

    HEaaN::Ciphertext ctxt_temp(context);

    if (length >2) {
        for (int i = 1; i < length-1; ++i) {
            eval.mult(basis[i], polynomial[i], ctxt_temp);
            eval.add(ctxt_out, ctxt_temp, ctxt_out);
        }
    }

    eval.add(ctxt_out, polynomial[0], ctxt_out);

    return ctxt_out;
}


// For vector slicing. slice vector from a_index to b_index
std::vector<double> vectorSlice(const std::vector<double> &input, int a, int b) {
    auto first = input.cbegin() + a;
    auto last = input.cbegin() + b;
    return std::vector<double>(first, last);
}


// GiantStep algorithm in Han-Ki
HEaaN::Ciphertext GiantStep(HEaaN::Context context, HEaaN::HomEvaluator eval, 
                            const std::vector<HEaaN::Ciphertext> &BS_basis, 
                            const std::vector<HEaaN::Ciphertext> &GS_basis, 
                            const std::vector<double> &polynomial, 
                            int k, int l) {

    HEaaN::Ciphertext ctxt_result(context);

    if (polynomial.size() < k) {
        ctxt_result = BabyStep(context, eval, BS_basis, polynomial, k);
        return ctxt_result;
    }

    // integer exp s.t. 2^exp <= n/k < 2^(exp+1)

    int exp = (int)floor(log2((double)polynomial.size()/(double)k));

    std::vector<double> quotient = vectorSlice(polynomial, k*pow(2,exp), polynomial.size());
    std::vector<double> remainder = vectorSlice(polynomial, 0, k*pow(2,exp));

    HEaaN::Ciphertext ctxt_quotient(context);
    HEaaN::Ciphertext ctxt_remainder(context);

    ctxt_quotient = GiantStep(context, eval, BS_basis, GS_basis, quotient, k, l);
    ctxt_remainder = GiantStep(context, eval, BS_basis, GS_basis, remainder, k, l);

    HEaaN::Ciphertext ctxt_temp(context);
    eval.mult(ctxt_quotient, GS_basis[exp], ctxt_temp);
    eval.add(ctxt_temp, ctxt_remainder, ctxt_result);

    return ctxt_result;
}

// Evaluating poly by using BSGS.
HEaaN::Ciphertext evalPolynomial(HEaaN::Context context, HEaaN::HomEvaluator eval, 
                                HEaaN::Ciphertext ctxt, const std::vector<double> &polynomial) {
                                    
    int m = ceil(log2(polynomial.size() + 1));
    int a = m / 2;

    int k = pow(2,a);
    int l = m -a;

    HEaaN::Ciphertext ctxt_trash(context);

    std::vector<HEaaN::Ciphertext> BS_basis;
    std::vector<HEaaN::Ciphertext> GS_basis;

    std::cout << "SetUp begin ..." << std::endl << std:endl;
    SetUp(context, eval, ctxt, BS_basis, GS_basis, k, l);
    HEaaN::Ciphertext ctxt_out(context);
    std::cout << "done" << std::endl << std:endl;  

    std::cout << "Giant Step begin ..." << std::endl << std:endl;
    ctxt_out = GiantStep(context, eval, BS_basis, GS_basis, polynomial, k, l);

    return ctxt_out;
}

//Aproximated ReLU function.
HEaaN::Ciphertext ApproxReLU(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext ctxt) {

    //imaginary removal BTS
    HEaaN::Ciphertext ctxt_temp(context);
    eval.conjugate(ctxt, ctxt_temp);
    eval.add(ctxt_temp, ctxt, ctxt_temp);
    //eval.mult(ctxt_temp, 0.5, ctxt_temp);
    //instead, we apply 0.5*p1 after imaginary removal BTS


    std::cout << "Result ciphertext - level " << ctxt_temp.getLevel()
                  << std::endl
                  << std::endl;
    
    HEaaN::Ciphertext ctxt_real_BTS(context);
    std::cout << "Imaginary Removal Bootstrapping ... " << std::endl;

    eval.bootstrap(ctxt_temp, ctxt_real_BTS, true);

    std::cout << "Result ciphertext after imaginary removal bootstrapping - level "
                  << ctxt_real_BTS.getLevel() << std::endl
                  << std::endl;

    const std::vector<double> &polynomial_1 = {
    -3.38572283433492e-47, 2.49052143193754e01 , 7.67064296707865e-45,
    -6.82383057582430e02 ,-1.33318527258859e-43, 6.80942845390599e03 ,
     9.19464568002043e-43,-3.12507100017105e04 ,-3.02547883089949e-42,
     7.47659388363757e04 , 5.02426027571770e-42,-9.65076838475839e04 ,
    -4.05931240321443e-42, 6.36977923778246e04 , 1.26671427827897e-42,
    -1.68602621347190e04   };

    // product 1/2 to coeff of p_1 because we need to calculate 1/2(x+bar(x)) to do BTS.
    for (int i = 0; i < polynomial_1.size(); ++i) {
        polynomial_1[i] = polynomial_1[i] * 0.5;
    }

    const std::vector<double> &polynomial_2 = {
    -9.27991756967991e-46, 1.68285511926011e01  , 8.32408114686671e-44,
    -3.39811750495659e02,-1.27756566628511e-42 , 2.79069998793847e03 ,
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

    std::cout << "first polynomial evaluation ... " << std::endl;
    ctxt_temp = evalPolynomial(context, eval, ctxt_real_BTS, polynomial_1);
    std::cout << "done" << std::endl << std:endl;


    std::cout << "second polynomial evaluation ... " << std::endl;

    HEaaN::Ciphertext ctxt_result(context);
    ctxt_result = evalPolynomial(context, eval, ctxt_temp2, polynomial_2);

    std::cout << "done" << std::endl << std:endl;



    std::cout << "Result ciphertext after evaluating p1 and p2 - level " << ctxt_result.getLevel()
                  << std::endl
                  << std::endl;
    
    std::cout << "Bootstrapping ... " << std::endl;

    HEaaN::Ciphertext ctxt_BTS(context);
    eval.bootstrap(ctxt_result, ctxt_BTS, true);

    std::cout << "Result ciphertext after bootstrapping - level "
                  << ctxt_BTS.getLevel() << std::endl
                  << std::endl;

    //ctxt_out3 = eval.sign(ctxt)
    
    ctxt_result = evalPolynomial(context, eval, ctxt_BTS, polynomial_3);

    //ReLU(x)= 0.5(x+x*sign(x))
    eval.mult(ctxt_real_BTS, ctxt_result, ctxt_result);
    eval.add(ctxt_real_BTS, ctxt_result, ctxt_result);
    eval.mult(ctxt_result, 0.5, ctxt_result);

    return ctxt_result;
}
