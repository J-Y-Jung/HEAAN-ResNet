#pragma once
#include <iostream>
#include <cmath>
#include "oddLazyBSGS.hpp"
#include "HEaaN/heaan.hpp"

namespace {
    using namespace HEaaN;
    using namespace std;
}

void oddGiantStep1(HEaaN::Context context, HEaaN::HomEvaluator eval,
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
        return;
    }

    else {
        int deg_div = k * pow(2, a);

        std::vector<double> quotient = vectorSlice(polynomial, deg_div, polynomial.size());
        std::vector<double> remainder = vectorSlice(polynomial, 0, deg_div);

        HEaaN::Ciphertext ctxt_quotient(context);
        HEaaN::Ciphertext ctxt_remainder(context);

        oddGiantStep1(context, eval, oddBS_basis, GS_basis, quotient, ctxt_quotient, k, l);
        oddGiantStep1(context, eval, oddBS_basis, GS_basis, remainder, ctxt_remainder, k, l);

        eval.add(ctxt_result, ctxt_remainder, ctxt_result);

    }

}


// Evaluating poly by using BSGS.
void evaloddPolynomial1(HEaaN::Context context, HEaaN::HomEvaluator eval,
    HEaaN::Ciphertext& ctxt, HEaaN::Ciphertext& ctxt_poly,
    const std::vector<double>& polynomial, int k, int l) {

    std::vector<HEaaN::Ciphertext> oddBS_basis;
    std::vector<HEaaN::Ciphertext> evenBS_basis;
    std::vector<HEaaN::Ciphertext> GS_basis;

    oddSetUp(context, eval, ctxt, oddBS_basis, evenBS_basis, GS_basis, k, l);

    evenBS_basis.clear();
    evenBS_basis.shrink_to_fit();

    HEaaN::Ciphertext ctxt_temp(context);
    oddGiantStep1(context, eval, oddBS_basis, GS_basis, polynomial, ctxt_temp, k, l);

}

//Aproximated ReLU function.
void ApproxReLU1(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext& ctxt, HEaaN::Ciphertext& ctxt_relu) {
    //cout << "input ctxt level = " << ctxt.getLevel() <<"\n";
    HEaaN::Ciphertext ctxt_temp(context);
    eval.conjugate(ctxt, ctxt_temp);
    eval.add(ctxt_temp, ctxt, ctxt_temp);
    eval.mult(ctxt_temp, 0.5, ctxt_temp);
    
    //cout << "after imaginary removal = " << ctxt_temp.getLevel() << "\n";

    
    HEaaN::Ciphertext ctxt_real_BTS(context);
    eval.bootstrap(ctxt_temp, ctxt_real_BTS, true);
    
    //cout << "after 1st BTS = " << ctxt_real_BTS.getLevel() << "\n";


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

    evalOddPolynomial(context, eval, ctxt_real_BTS, ctxt_temp, polynomial_1, 4, 2);
    
    //cout << "after 1st poly eval = " << ctxt_temp.getLevel() << "\n";

    HEaaN::Ciphertext ctxt_temp1(context);
    evalOddPolynomial(context, eval, ctxt_temp, ctxt_temp1, polynomial_2, 2, 3);
 
    
    //cout << "after 2nd poly eval = " << ctxt_temp1.getLevel() << "\n";
    eval.bootstrap(ctxt_temp1, ctxt_temp, true);
    eval.levelDown(ctxt_temp, 7, ctxt_temp);
    
    //cout << "after 2nd BTS = " << ctxt_temp.getLevel() << "\n";
    evalOddPolynomial(context, eval, ctxt_temp, ctxt_temp1, polynomial_3, 4, 3);
    
    //cout << "after 3rd poly eval = " << ctxt_temp1.getLevel() << "\n";
    
    eval.mult(ctxt_real_BTS, 0.5, ctxt_temp);
    eval.mult(ctxt_real_BTS, ctxt_temp1, ctxt_relu);
    eval.add(ctxt_temp, ctxt_relu, ctxt_relu);
    
    //cout << "final level = " << ctxt_relu.getLevel() << "\n";
    
    return;
    
}
