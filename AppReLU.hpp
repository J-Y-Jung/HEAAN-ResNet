﻿#pragma once
#pragma once
#include <iostream>
#include <cmath>
#include "HEaaN/heaan.hpp"

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
void AppReLU(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext& ctxt, HEaaN::Ciphertext& ctxt_relu) {

    // std::cout << "Input ciphertext - level " << ctxt.getLevel()
    //     << std::endl
    //     << std::endl;

    HEaaN::Ciphertext ctxt_temp(context);
    eval.conjugate(ctxt, ctxt_temp);
    eval.add(ctxt_temp, ctxt, ctxt_temp);
    eval.mult(ctxt_temp, 0.5, ctxt_temp);

    HEaaN::Ciphertext ctxt_real_BTS(context);
    //     std::cout << "Imaginary Removal Bootstrapping ... " << std::endl;

    eval.bootstrap(ctxt_temp, ctxt_real_BTS, true);

    /*std::cout << "Result ciphertext after imaginary removal bootstrapping - level "
        << ctxt_real_BTS.getLevel() << std::endl
        << std::endl;*/

    std::vector<double> polynomial_1 = { 
                     3.85169741234183e-44, 1.80966285718807e1, -4.59730416916377e-42, -4.34038703274886e2, 
                     7.96299160375690e-41, 4.15497103545696e3, -5.28977110396316e-40, -1.86846943613149e4, 
                     1.67219551148917e-39, 4.41657177889329e4, -2.69777424798506e-39, -5.65527928983401e4, 
                     2.14124591383569e-39, 3.71156122725781e4, -6.61722455927198e-40, -9.78241933892781e3 };

    /*
    // product 1/2 to coeff of p_2 because we need to calculate 1/2(x+bar(x)) to do BTS.
    for(int  i = 0 ; i < polynomial_1.size() ; ++i){
        polynomial_1[i] = polynomial_1[i]* 0.5;
    }
    */

    std::vector<double> polynomial_2 = {
                -1.04501074063854e-46 , 3.79753323360856  , 4.22842209818016e-45, -1.17718157771192e1,
                -2.25571113936639e-44 , 2.49771086678346e1, 4.42462875106862e-44, -3.15238841603993e1,
                -4.13554194411645e-44 , 2.37294863126722e1, 2.00060158783094e-44, -1.04331800195923e1,
                -4.86041132712796e-45 , 2.46743976260838  , 4.71256214052049e-46, -2.42130100247617e-1 };

    /*
        // product 1/2 to coeff of p_2 because we need to calculate 1/2(x+bar(x)) to do BTS.
        for(int  i = 0 ; i < polynomial_2.size() ; ++i){
            polynomial_2[i] = polynomial_2[i]*0.5;
        }
    
    std::vector<double> polynomial_3 = {
    6.43551938319983e-48, 5.07813569758861, 8.12601038855762e-46, -3.07329918137186e1,
    -1.60198474678427e-44, 1.44109746812809e2, 1.07463154460511e-43, -4.59661688826142e2,
    -3.63448723044512e-43, 1.02152064470459e3, 7.25207125369784e-43, -1.62056256708877e3,
    -9.27306397853655e-43, 1.86467646416570e3, 7.95843097354065e-43, -1.56749300877143e3,
    -4.69190103147527e-43, 9.60970309093422e2, 1.90863349654016e-43, -4.24326161871646e2,
    -5.27439678020696e-44, 1.31278509256003e2, 9.47044937974786e-45, -2.69812576626115e1,
    -9.98181561763750e-46, 3.30651387315565, 4.69390466192199e-47, -1.82742944627533e-1
    };*/


    //for optimization
    for (int i = 0; i < 16 ; ++i) {
        polynomial_2[i] = polynomial_2[i] * 0.5;
    }


    //HEaaN::Ciphertext ctxt_temp(context);

    /*std::cout << "1st polynomial evaluation ... " << std::endl << std::endl;*/
    evalOddPolynomial(context, eval, ctxt_real_BTS, ctxt_temp, polynomial_1, 4, 2);
    /*std::cout << "done" << std::endl << std::endl;

    std::cout << "Ciphertext after evaluating 1st polynomial - level " << ctxt_temp1.getLevel()
        << std::endl
        << std::endl;

    std::cout << "2nd polynomial evaluation ... " << std::endl << std::endl;*/

    HEaaN::Ciphertext ctxt_temp1(context);
    evalOddPolynomial(context, eval, ctxt_temp, ctxt_temp1, polynomial_2, 2, 3);


    eval.mult(ctxt_real_BTS, 0.5, ctxt_temp);
    eval.mult(ctxt_real_BTS, ctxt_temp1, ctxt_relu);
    eval.add(ctxt_temp, ctxt_relu, ctxt_relu);


    //eval.mult(ctxt_real_BTS, 0.5, ctxt_real_BTS);
    //eval.mult(ctxt_real_BTS, ctxt_sign, ctxt_relu);
    //eval.add(ctxt_real_BTS, ctxt_relu, ctxt_relu);

    // std::cout << "Evaluating Apporximate ReLU done; Ciphertext after evaluating approximate ReLU - level " << ctxt_relu.getLevel()
    //     << std::endl
    //     << std::endl;
}
