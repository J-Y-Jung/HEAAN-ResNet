#pragma once
#include <iostream>
#include <cmath>
#include "HEaaN/heaan.hpp"

void print_polynomial(const std::vector<double> &polynomial, const size_t degree) {
    auto print_coeff = [](double coeff) {
        if (coeff < 0.0L)
            std::cout << " - " << std::abs(coeff);
        else
            std::cout << " + " << coeff;
    };

    std::cout << polynomial[0] << "X^" << degree;
    for (size_t idx = 1 ; idx < degree-1 ; ++idx) {
        print_coeff(polynomial[idx]);
        std::cout << "X^" << degree-idx;
    }
    print_coeff(polynomial[degree-1]);
    std::cout << "X";
    print_coeff(polynomial[degree]);
    
    std::cout << std::endl;
}



// Construct BabyStep basis and GiantStep basis.
void SetUp(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext& ctxt,
    std::vector<HEaaN::Ciphertext>& BS_basis,
    std::vector<HEaaN::Ciphertext>& GS_basis,
    const int k, const int l) {

    //first, make BS_basis={x, x, ..., x} of length=k+1
    for (int i = 0; i < k + 1; ++i) {
        BS_basis.push_back(ctxt);
    }

    HEaaN::Ciphertext ctxt_temp(context);

    for (int i = 2; i < k + 1; i = 2 * i) {
        eval.mult(BS_basis[i / 2], BS_basis[i / 2], ctxt_temp);
        BS_basis[i] = ctxt_temp;
    }

    for (int i = 2; i < k + 1; ++i) {

        int alpha = (int)floor(log2(i));
        int ind = pow(2, alpha);

        if (ind < i) {
            eval.mult(BS_basis[ind], BS_basis[i - ind], ctxt_temp);
            BS_basis[i] = ctxt_temp;
        }
    }
    
    //first element of GS_basis is x^k
    GS_basis.push_back(BS_basis[k]);
    BS_basis.pop_back();

    for (int j = 1; j < l; ++j) {
        eval.mult(GS_basis[j - 1], GS_basis[j - 1], ctxt_temp);
        GS_basis.push_back(ctxt_temp);
    }
    
}


std::vector<double> vectorSlice(const std::vector<double> &input, int a, int b) {
    auto first = input.begin() + a;
    auto last = input.begin() + b;
    return std::vector<double>(first, last);
}

void linearMult(HEaaN::Context context, HEaaN::HomEvaluator eval,
            const std::vector<HEaaN::Ciphertext> &basis,
            const std::vector<double> &polynomial,
            HEaaN::Ciphertext &ctxt_result){
    eval.mult(basis[1],polynomial[1],ctxt_result);
    eval.add(ctxt_result,polynomial[0],ctxt_result);

    }


void GiantStep(HEaaN::Context context, HEaaN::HomEvaluator eval, 
            const std::vector<HEaaN::Ciphertext> &BS_basis, 
            const std::vector<HEaaN::Ciphertext> &GS_basis, 
            const std::vector<double> &polynomial,
            HEaaN::Ciphertext &ctxt_result,
            int k, int l,int value){

    int a = value/2; // value = k*pow(2,l)
    
    if(a==1){
        linearMult(context,eval,BS_basis,polynomial,ctxt_result);
        return;
    }else{
        if(polynomial.size() - a != 0){
            std::vector<double> quotient = vectorSlice(polynomial , a, polynomial.size());
            std::vector<double> remainder = vectorSlice(polynomial , 0 , a);

            HEaaN::Ciphertext ctxt_quotient(context);
            HEaaN::Ciphertext ctxt_remainder(context);

            GiantStep(context, eval, BS_basis, GS_basis, quotient, ctxt_quotient, k, l , a);
            GiantStep(context, eval, BS_basis, GS_basis, remainder, ctxt_remainder, k, l ,a);

            HEaaN::Ciphertext ctxt_temp(context);
            if(a>=4){
                eval.mult(ctxt_quotient , GS_basis[log2(a/4)], ctxt_temp);
                eval.add(ctxt_temp,ctxt_remainder,ctxt_result);
            }else{
                eval.mult(ctxt_quotient, BS_basis[a],ctxt_temp);
                eval.add(ctxt_temp, ctxt_remainder, ctxt_result);
            }
        }else{
            //this case measn that quotient part does not need in division algo.
            std::vector<double> remainder = vectorSlice(polynomial , 0 , a);
            
            HEaaN::Ciphertext ctxt_remainder(context);
            GiantStep(context, eval, BS_basis, GS_basis, remainder, ctxt_result, k, l ,a);
        }

    }
}


// Evaluating poly by using BSGS.
void evalPolynomial(HEaaN::Context context, HEaaN::HomEvaluator eval, 
                                HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_poly,
                                const std::vector<double> &polynomial) {
                                    
    int m = ceil(log2(polynomial.size()));
    int a = m / 2;
    
    int k = pow(2,a);
    int l = m - a;

    std::vector<HEaaN::Ciphertext> BS_basis;
    std::vector<HEaaN::Ciphertext> GS_basis;

    SetUp(context, eval, ctxt, BS_basis, GS_basis, k, l);
    GiantStep(context, eval, BS_basis, GS_basis, polynomial, ctxt_poly, k, l, k*pow(2,l));
}

//Aproximated ReLU function.
void ApproxReLU(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext &ctxt, HEaaN::Ciphertext &ctxt_relu) {
    
    HEaaN::Ciphertext ctxt_temp(context);
    eval.conjugate(ctxt, ctxt_temp);
    eval.add(ctxt_temp, ctxt, ctxt_temp);
    eval.mult(ctxt_temp, 0.5, ctxt_temp);
   
    HEaaN::Ciphertext ctxt_real_BTS(context);
    std::cout << "Imaginary Removal Bootstrapping ... " << std::endl;

    eval.bootstrap(ctxt_temp, ctxt_real_BTS, true);

    std::cout << "Result ciphertext after imaginary removal bootstrapping - level "
                  << ctxt_real_BTS.getLevel() << std::endl
                  << std::endl;

    std::vector<double> polynomial_1 = {
     1.34595769293910e-33, 2.45589415425004e1, 4.85095667238242e-32, -6.69660449716894e2, 
    -2.44541235853840e-30, 6.67299848301339e3, 1.86874811944640e-29, -3.06036656163898e4,
    -5.76227817577242e-29, 7.31884032987787e4, 8.53680673009259e-29, -9.44433217050084e4, 
    -6.02701474694667e-29, 6.23254094212546e4, 1.62342843661940e-29, -1.64946744117805e4 
    };

    std::vector<double> polynomial_2 = {
    1.53261588585630e-47, 9.35625636035439e0, -3.68972123048249e-46, -5.91638963933626e1, 
    1.74254399703303e-45, 1.48860930626448e2, -3.20672110002213e-45, -1.75812874878582e2, 
    2.79115738948645e-45, 1.09111299685955e2, -1.22590309306100e-45, -3.66768839978755e1, 
    2.62189142557962e-46, 6.31846290311294e0, -2.16662326421275e-47, -4.37113415082177e-01
    };
    

    std::vector<double> polynomial_3 = {
     6.43551938319983e-48, 5.07813569758861e0, 8.12601038855762e-46, -3.07329918137186e1,
    -1.60198474678427e-44, 1.44109746812809e2, 1.07463154460511e-43, -4.59661688826142e2,
    -3.63448723044512e-43, 1.02152064470459e3, 7.25207125369784e-43, -1.62056256708877e3,
    -9.27306397853655e-43, 1.86467646416570e3, 7.95843097354065e-43, -1.56749300877143e3,
    -4.69190103147527e-43, 9.60970309093422e2, 1.90863349654016e-43, -4.24326161871646e2,
    -5.27439678020696e-44, 1.31278509256003e2, 9.47044937974786e-45, -2.69812576626115e1,
    -9.98181561763750e-46, 3.30651387315565e0, 4.69390466192199e-47, -1.82742944627533e-1
    };
    

    HEaaN::Ciphertext ctxt_temp1(context);

    std::cout << "first polynomial evaluation ... " << std::endl << std::endl;
    evalPolynomial(context, eval, ctxt_real_BTS, ctxt_temp1, polynomial_1);
    std::cout << "done" << std::endl << std::endl;
    
    std::cout << "Ciphertext after evaluating poly1 - level " << ctxt_temp1.getLevel()
                  << std::endl
                  << std::endl;

    
    std::cout << "second polynomial evaluation ... " << std::endl << std::endl;

    
    HEaaN::Ciphertext ctxt_temp2(context);
    evalPolynomial(context, eval, ctxt_temp1, ctxt_temp2, polynomial_2);
    std::cout << "done" << std::endl << std::endl;

    
    std::cout << "Ciphertext after evaluating poly2 - level " << ctxt_temp2.getLevel()
                  << std::endl
                  << std::endl;
    
    std::cout << "Bootstrapping ... " << std::endl;
    HEaaN::Ciphertext ctxt_BTS(context);
    eval.bootstrap(ctxt_temp2, ctxt_BTS, true);

    std::cout << "Result ciphertext after bootstrapping - level "
                  << ctxt_BTS.getLevel() << std::endl
                  << std::endl;

    
    std::cout << "third polynomial evaluation ... " << std::endl << std::endl;

    HEaaN::Ciphertext ctxt_sign(context);
    evalPolynomial(context, eval, ctxt_BTS, ctxt_sign, polynomial_3);
     std::cout << "Ciphertext after evaluating poly3 - level " << ctxt_sign.getLevel()
                  << std::endl
                  << std::endl;

    HEaaN::Ciphertext ctxt_scalar(context);
    //mult 0.5 to x . ReLU = 0.5x + 0.5x*sign(x)
    eval.mult(ctxt_real_BTS, 0.5, ctxt_scalar);
    eval.mult(ctxt_scalar, ctxt_sign, ctxt_relu);
    eval.add(ctxt_scalar, ctxt_relu, ctxt_relu);
    

    std::cout << "Evaluating Apporximate ReLU done " << std::endl;
    
}
