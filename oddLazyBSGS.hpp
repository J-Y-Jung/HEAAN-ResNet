#pragma once
#include <iostream>
#include <cmath>
#include "HEaaN/heaan.hpp"
#include <time.h>

void multWithoutRelin(HEaaN::Context context, HEaaN::HomEvaluator eval,
    const HEaaN::Ciphertext& ctxt1, const HEaaN::Ciphertext& ctxt2, HEaaN::Ciphertext& ctxt_out) {

    const HEaaN::u64 level1 = ctxt1.getLevel();
    const HEaaN::u64 level2 = ctxt2.getLevel();

    if (level1 == level2) {
        eval.tensor(ctxt1, ctxt2, ctxt_out);
        eval.rescale(ctxt_out);
        return;
    }

    HEaaN::Ciphertext ctxt_tmp(context);

    if (level1 > level2) {
        eval.levelDown(ctxt1, level2, ctxt_tmp);
        eval.tensor(ctxt_tmp, ctxt2, ctxt_out);
        eval.rescale(ctxt_out);
    }
    else { // level1 < level2
        eval.levelDown(ctxt2, level1, ctxt_tmp);
        eval.tensor(ctxt1, ctxt_tmp, ctxt_out);
        eval.rescale(ctxt_out);
    }
}

// Construct BabyStep basis and GiantStep basis.
void oddSetUp(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext& ctxt,
    std::vector<HEaaN::Ciphertext>& oddBS_basis,
    std::vector<HEaaN::Ciphertext>& evenBS_basis,
    std::vector<HEaaN::Ciphertext>& GS_basis,
    const int k, const int l) {

    oddBS_basis.push_back(ctxt);
    evenBS_basis.push_back(ctxt);

    HEaaN::Ciphertext ctxt_temp(context);

    for (int i = 1; i <= log2(k); ++i) {
        eval.mult(evenBS_basis[i - 1], evenBS_basis[i - 1], ctxt_temp);
        evenBS_basis.push_back(ctxt_temp);
    }


    for (int i = 1; i < k / 2; ++i) {

        int alpha = (int)floor(log2(2 * i + 1));
        int ind = pow(2, alpha - 1);

        eval.mult(evenBS_basis[alpha], oddBS_basis[i - ind], ctxt_temp);
        oddBS_basis.push_back(ctxt_temp);
    }

    //first element of GS_basis is x^k
    GS_basis.push_back(evenBS_basis[log2(k)]);
    /*
    for (int i = 0; i < oddBS_basis.size(); ++i) {
        std::cout << i << "th Ciphertext of oddBS_basis vector  - level "
            << oddBS_basis[i].getLevel() << std::endl
            << std::endl;
    }
    */
    for (int j = 1; j < l; ++j) {
        eval.mult(GS_basis[j - 1], GS_basis[j - 1], ctxt_temp);
        GS_basis.push_back(ctxt_temp);
    }
    /*
    for (int j = 0; j < GS_basis.size(); ++j) {
        std::cout << j << "th Ciphertext of GS_basis vector  - level "
            << GS_basis[j].getLevel() << std::endl
            << std::endl;
    }
    */
}

//BabyStep algo in Han-Ki.
void oddBabyStep(HEaaN::Context context, HEaaN::HomEvaluator eval,
    const std::vector<HEaaN::Ciphertext>& oddBS_basis,
    const std::vector<double>& polynomial,
    HEaaN::Ciphertext& ctxt_result,
    const int k) {

    eval.mult(oddBS_basis[0], polynomial[1], ctxt_result);
    /*
    std::cout << "Current Ciphertext initial part of odd Baby Step  - level "
        << ctxt_result.getLevel() << std::endl
        << std::endl;
        */

    HEaaN::Ciphertext ctxt_temp(context);

    if (k > 2) {
        for (int i = 1; i < k / 2; ++i) {
            eval.mult(oddBS_basis[i], polynomial[2 * i + 1], ctxt_temp);
            eval.add(ctxt_result, ctxt_temp, ctxt_result);
            /*
            std::cout << "Current Ciphertext during odd Baby Step  - level "
                << ctxt_result.getLevel() << std::endl
                << std::endl;
                */
        }
    }
}


// For vector slicing. slice vector from a_index to b_index
std::vector<double> vectorSlice(const std::vector<double>& input, int a, int b) {
    auto first = input.begin() + a;
    auto last = input.begin() + b;
    return std::vector<double>(first, last);
}


// GiantStep algorithm in Han-Ki
void oddGiantStep(HEaaN::Context context, HEaaN::HomEvaluator eval,
    const std::vector<HEaaN::Ciphertext>& oddBS_basis,
    const std::vector<HEaaN::Ciphertext>& GS_basis,
    const std::vector<double>& polynomial,
    HEaaN::Ciphertext& ctxt_result,
    int k, int l) {

    //integer a s.t. 2^a <= n/k < 2^(a+1)
    int degree = polynomial.size() - 1;

    int a = (int)floor(log2((double)degree / (double)k));

    if (a < 0) {
        oddBabyStep(context, eval, oddBS_basis, polynomial, ctxt_result, k);
        /*
        std::cout << "Current Ciphertext evaluating one-time odd Baby Step  - level "
            << ctxt_result.getLevel() << std::endl
            << std::endl;
            */
        return;
    }

    else {
        int deg_div = k * pow(2, a);

        std::vector<double> quotient = vectorSlice(polynomial, deg_div, polynomial.size());
        std::vector<double> remainder = vectorSlice(polynomial, 0, deg_div);

        HEaaN::Ciphertext ctxt_quotient_relin(context);
        HEaaN::Ciphertext ctxt_remainder(context);

        if (quotient.size() <= k) {
            oddGiantStep(context, eval, oddBS_basis, GS_basis, quotient, ctxt_quotient_relin, k, l);
            /*
            std::cout << "Current Ciphertext evaluating the last quotient  - level "
                << ctxt_quotient_relin.getLevel() << std::endl
                << std::endl;
                */

        }

        else {

            HEaaN::Ciphertext ctxt_quotient(context);

            oddGiantStep(context, eval, oddBS_basis, GS_basis, quotient, ctxt_quotient, k, l);
            /*
            std::cout << "Current Ciphertext evaluating quotient before relinearize - level "
                << ctxt_quotient.getLevel() << std::endl
                << std::endl;
            */
            eval.relinearize(ctxt_quotient, ctxt_quotient_relin);
            /*
            std::cout << "Current Ciphertext evaluating quotient after relinearize - level "
                << ctxt_quotient_relin.getLevel() << std::endl
                << std::endl;
                */
        }

        oddGiantStep(context, eval, oddBS_basis, GS_basis, remainder, ctxt_remainder, k, l);
        /*
        std::cout << "Current Ciphertext evaluating remainder  - level "
            << ctxt_remainder.getLevel() << std::endl
            << std::endl;
            */
        multWithoutRelin(context, eval, GS_basis[a], ctxt_quotient_relin, ctxt_result);
        eval.add(ctxt_result, ctxt_remainder, ctxt_result);
        /*
        std::cout << "Current Ciphertext evaluating one-time Giant Step  - level "
            << ctxt_result.getLevel() << std::endl
            << std::endl;
            */
    }

}


// Evaluating poly by using BSGS.
void evalOddPolynomial(HEaaN::Context context, HEaaN::HomEvaluator eval,
    HEaaN::Ciphertext& ctxt, HEaaN::Ciphertext& ctxt_poly,
    const std::vector<double>& polynomial, int k, int l) {

    std::vector<HEaaN::Ciphertext> oddBS_basis;
    std::vector<HEaaN::Ciphertext> evenBS_basis;
    std::vector<HEaaN::Ciphertext> GS_basis;

    oddSetUp(context, eval, ctxt, oddBS_basis, evenBS_basis, GS_basis, k, l);

    HEaaN::Ciphertext ctxt_temp(context);
    oddGiantStep(context, eval, oddBS_basis, GS_basis, polynomial, ctxt_temp, k, l);

    oddBS_basis.clear();
    evenBS_basis.shrink_to_fit();

    eval.relinearize(ctxt_temp, ctxt_poly);

}

//Aproximated ReLU function.
void ApproxReLU(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext& ctxt, HEaaN::Ciphertext& ctxt_relu, double polytime, double btstime, double etctime) {
    
    clock_t start1, start2, start3, start4, start5, start6, start7, end1, end2, end3, end4, end5, end6, end7;
    
    double result1;
    start1 = clock();
    HEaaN::Ciphertext ctxt_temp(context);
    eval.conjugate(ctxt, ctxt_temp);
    eval.add(ctxt_temp, ctxt, ctxt_temp);
    eval.mult(ctxt_temp, 0.5, ctxt_temp);
    end1 = clock();
    result1 = (double)(end1 - start1);
    
    
    double result2;
    start2 = clock();
    HEaaN::Ciphertext ctxt_real_BTS(context);
    eval.bootstrap(ctxt_temp, ctxt_real_BTS, true);
    end2 = clock();
    result2 = (double)(end2 - start2);
    

    std::vector<double> polynomial_1 = {
    1.34595769293910e-33, 2.45589415425004e1, 4.85095667238242e-32, -6.69660449716894e2,
    -2.44541235853840e-30, 6.67299848301339e3, 1.86874811944640e-29, -3.06036656163898e4,
    -5.76227817577242e-29, 7.31884032987787e4, 8.53680673009259e-29, -9.44433217050084e4,
    -6.02701474694667e-29, 6.23254094212546e4, 1.62342843661940e-29, -1.64946744117805e4
    };

    std::vector<double> polynomial_2 = {
    1.53261588585630e-47, 9.35625636035439, -3.68972123048249e-46, -5.91638963933626e1,
    1.74254399703303e-45 , 1.48860930626448e2, -3.20672110002213e-45, -1.75812874878582e2,
    2.79115738948645e-45, 1.09111299685955e2, -1.22590309306100e-45, -3.66768839978755e1,
    2.62189142557962e-46, 6.31846290311294, -2.16662326421275e-47, -4.37113415082177e-01
    };

    std::vector<double> polynomial_3 = {
    6.43551938319983e-48, 5.07813569758861, 8.12601038855762e-46, -3.07329918137186e1,
    -1.60198474678427e-44, 1.44109746812809e2, 1.07463154460511e-43, -4.59661688826142e2,
    -3.63448723044512e-43, 1.02152064470459e3, 7.25207125369784e-43, -1.62056256708877e3,
    -9.27306397853655e-43, 1.86467646416570e3, 7.95843097354065e-43, -1.56749300877143e3,
    -4.69190103147527e-43, 9.60970309093422e2, 1.90863349654016e-43, -4.24326161871646e2,
    -5.27439678020696e-44, 1.31278509256003e2, 9.47044937974786e-45, -2.69812576626115e1,
    -9.98181561763750e-46, 3.30651387315565, 4.69390466192199e-47, -1.82742944627533e-1
    };
    
    
    //for optimization
    for(int  i = 0 ; i < 28 ; ++i){
        polynomial_3[i] = polynomial_3[i] * 0.5;
    }



    double result3;
    start3 = clock();
    evalOddPolynomial(context, eval, ctxt_real_BTS, ctxt_temp, polynomial_1, 4, 2);
    end3 = clock();
    
    
    double result4;
    start4 = clock();
    HEaaN::Ciphertext ctxt_temp1(context);
    evalOddPolynomial(context, eval, ctxt_temp, ctxt_temp1, polynomial_2, 2, 3);
    end4 = clock();
    result4 = (double)(end4 - start4);
    
    
    double result5;
    start5 = clock();
    eval.bootstrap(ctxt_temp1, ctxt_temp, true);
    end5 = clock();
    result5 = (double)(end5 - start5);
    
    
    double result6;
    start6 = clock();
    evalOddPolynomial(context, eval, ctxt_temp, ctxt_temp1, polynomial_3, 4, 3);
    end6 = clock();
    result6 = (double)(end6 - start6);
    
    
    double result7;
    start7 = clock();
    eval.mult(ctxt_real_BTS, 0.5, ctxt_temp);
    eval.mult(ctxt_real_BTS, ctxt_temp1, ctxt_relu);
    eval.add(ctxt_temp, ctxt_relu, ctxt_relu);
    
    eval.levelDown(ctxt_relu, 5, ctxt_relu);
    end7 = clock();
    result7 = (double)(end7-start7);
    
    polytime += result3 + result4 +result6;
    btstime += result2 + result5;
    etctime += result1 + result7;
    
}
