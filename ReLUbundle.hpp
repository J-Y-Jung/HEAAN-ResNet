#pragma once
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include "HEaaN/heaan.hpp"
#include "leveldown.hpp"

namespace{
    using namespace std;
    using namespace HEaaN;
}

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
  
    for (int j = 1; j < l; ++j) {
        eval.mult(GS_basis[j - 1], GS_basis[j - 1], ctxt_temp);
        GS_basis.push_back(ctxt_temp);
    }
   
}

//BabyStep algo in Han-Ki.
void oddBabyStep(HEaaN::Context context, HEaaN::HomEvaluator eval,
    const std::vector<HEaaN::Ciphertext>& oddBS_basis,
    const std::vector<double>& polynomial,
    HEaaN::Ciphertext& ctxt_result,
    const int k) {

    eval.mult(oddBS_basis[0], polynomial[1], ctxt_result);
  
    HEaaN::Ciphertext ctxt_temp(context);

    if (k > 2) {
        for (int i = 1; i < k / 2; ++i) {
            eval.mult(oddBS_basis[i], polynomial[2 * i + 1], ctxt_temp);
            eval.add(ctxt_result, ctxt_temp, ctxt_result);
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

        }

        else {

            HEaaN::Ciphertext ctxt_quotient(context);

            oddGiantStep(context, eval, oddBS_basis, GS_basis, quotient, ctxt_quotient, k, l);
          
            eval.relinearize(ctxt_quotient, ctxt_quotient_relin);
        }

        oddGiantStep(context, eval, oddBS_basis, GS_basis, remainder, ctxt_remainder, k, l);
      
        multWithoutRelin(context, eval, GS_basis[a], ctxt_quotient_relin, ctxt_result);
        eval.add(ctxt_result, ctxt_remainder, ctxt_result);
      
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

    eval.relinearize(ctxt_temp, ctxt_poly);

}





void ApproxReLU_bundle(HEaaN::Context context, HEaaN::KeyPack pack,HEaaN::HomEvaluator eval, 
    std::vector<std::vector<HEaaN::Ciphertext>>& ctxt_bundle, std::vector<std::vector<HEaaN::Ciphertext>>& ctxt_relu_bundle){
    
    HEaaN::Ciphertext ctxt_temp(context); //for initializing
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_temp_bundle(ctxt_relu_bundle.size() , std::vector<HEaaN::Ciphertext>(ctxt_relu_bundle[0].size(),ctxt_temp));
    
    //std::cout << "size1 = " << ctxt_bundle.size() << " size2 = " << ctxt_bundle[0].size() << std::endl;
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < ctxt_relu_bundle.size() ; i++){
        for(int j = 0 ; j < ctxt_relu_bundle[0].size() ; j++){
            eval.conjugate(ctxt_bundle[i][j],ctxt_temp_bundle[i][j]);
            eval.add(ctxt_temp_bundle[i][j],ctxt_bundle[i][j],ctxt_temp_bundle[i][j]);
            eval.mult(ctxt_temp_bundle[i][j],0.5,ctxt_temp_bundle[i][j]);
        }
    }

    //std::cout << "b" << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_real_BTS_bundle(ctxt_relu_bundle.size() , std::vector<HEaaN::Ciphertext>(ctxt_relu_bundle[0].size(),ctxt_temp));
    //#pragma omp parallel for
    //for(int i = 0 ; i < ctxt_relu_bundle.size() ; ++i){
    //    //#pragma omp parallel for
    //    for(int j = 0 ; j < ctxt_relu_bundle[0].size() ; ++j){
    //        eval.bootstrap(ctxt_temp_bundle[i][j],ctxt_real_BTS_bundle[i][j],true);
    //   }
    //}

    if(ctxt_relu_bundle.size() == 16 && ctxt_relu_bundle[0].size() == 16){
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //std::cout << "1" << std::endl;
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp_bundle[i/16][i%16],ctxt_real_BTS_bundle[i/16][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //std::cout << "2" << std::endl;
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp_bundle[(i/16)+3][i%16],ctxt_real_BTS_bundle[i/16+3][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //std::cout << "3" << std::endl;
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp_bundle[(i/16)+6][i%16],ctxt_real_BTS_bundle[i/16+6][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //std::cout <<"4"<<std::endl;
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp_bundle[(i/16)+9][i%16],ctxt_real_BTS_bundle[(i/16)+9][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //std::cout <<"5" << std::endl;
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp_bundle[(i/16)+12][i%16],ctxt_real_BTS_bundle[(i/16)+12][i%16],true);
        }

        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //std::cout << "6" << std::endl;
            eval.bootstrap(ctxt_temp_bundle[3*(i/8)+2][(i%8)+8],ctxt_real_BTS_bundle[3*(i/8)+2][(i%8)+8]);
        }
        #pragma omp parallel for num_threads(16)
        for(int i = 0 ; i < 16 ; i++){
            //std::cout << "7" << std::endl;
            eval.bootstrap(ctxt_temp_bundle[15][i%16],ctxt_real_BTS_bundle[15][i%16],true);
        }
    }else if(ctxt_relu_bundle.size() == 4 && ctxt_relu_bundle[0].size() == 32){
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp_bundle[0][i],ctxt_real_BTS_bundle[0][i],true);
        }
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp_bundle[1][i],ctxt_real_BTS_bundle[1][i],true);
        }
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp_bundle[2][i],ctxt_real_BTS_bundle[2][i],true);
        }
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp_bundle[3][i],ctxt_real_BTS_bundle[3][i],true);
        }
    }else if(ctxt_relu_bundle.size() == 8 && ctxt_relu_bundle[0].size() == 8){
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            eval.bootstrap(ctxt_temp_bundle[i/8][i%8],ctxt_real_BTS_bundle[i/8][i%8],true);
        }
        #pragma omp parallel for num_threads(24)
        for(int i = 0 ; i < 24 ; i++){
            eval.bootstrap(ctxt_temp_bundle[(i/8) + 5][i%8],ctxt_real_BTS_bundle[(i/8) + 5][i%8],true);
        }
    }else{
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            eval.bootstrap(ctxt_temp_bundle[0][i],ctxt_real_BTS_bundle[0][i],true);
        }
        #pragma omp parallel for num_threads(24)
        for(int i = 0 ; i < 24 ; i++){
            eval.bootstrap(ctxt_temp_bundle[0][i+40],ctxt_real_BTS_bundle[0][i+40],true);
        }
    }

    

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

    //std::cout << "c" << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_temp1_bundle(ctxt_relu_bundle.size() , std::vector<HEaaN::Ciphertext>(ctxt_relu_bundle[0].size(),ctxt_temp));
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < ctxt_temp1_bundle.size() ; ++i){
        for(int j = 0 ; j < ctxt_temp1_bundle[0].size() ; ++j){
            evalOddPolynomial(context,eval,ctxt_real_BTS_bundle[i][j],ctxt_temp1_bundle[i][j],polynomial_1,4,2);
        }
    }
    
    //std::cout << "d" << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_temp2_bundle(ctxt_relu_bundle.size() , std::vector<HEaaN::Ciphertext>(ctxt_relu_bundle[0].size(),ctxt_temp));
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < ctxt_temp2_bundle.size() ; ++i){
        for(int j = 0 ; j < ctxt_temp2_bundle[0].size() ; ++j){
            evalOddPolynomial(context,eval,ctxt_temp1_bundle[i][j],ctxt_temp2_bundle[i][j],polynomial_2,2,3);
        }
    }
    ctxt_temp1_bundle.clear();
    ctxt_temp1_bundle.shrink_to_fit();

    //BTS...
    //std::cout << "BTS..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_real_BTS2_bundle(ctxt_relu_bundle.size() , std::vector<HEaaN::Ciphertext>(ctxt_relu_bundle[0].size(),ctxt_temp));
    //#pragma omp parallel for collapse(2)
    //for(int i = 0 ; i < ctxt_real_BTS2_bundle.size() ; ++i){
    //    //#pragma omp parallel for
    //    for(int j = 0 ; j < ctxt_real_BTS2_bundle[0].size() ; ++j){
    //        eval.bootstrap(ctxt_temp2_bundle[i][j],ctxt_real_BTS2_bundle[i][j],true);
    //    }
    //}
    if(ctxt_relu_bundle.size() == 16 && ctxt_relu_bundle[0].size() == 16){
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp2_bundle[i/16][i%16],ctxt_real_BTS2_bundle[i/16][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp2_bundle[(i/16)+3][i%16],ctxt_real_BTS2_bundle[(i/16)+3][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp2_bundle[(i/16)+6][i%16],ctxt_real_BTS2_bundle[(i/16)+6][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp2_bundle[(i/16)+9][i%16],ctxt_real_BTS2_bundle[(i/16)+9][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            //#pragma omp parallel num_threads(1)
            eval.bootstrap(ctxt_temp2_bundle[(i/16)+12][i%16],ctxt_real_BTS2_bundle[(i/16)+12][i%16],true);
        }
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[3*(i/8)+2][(i%8)+8],ctxt_real_BTS2_bundle[3*(i/8)+2][(i%8)+8]);
        }
        #pragma omp parallel for num_threads(16)
        for(int i = 0 ; i < 16 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[15][i%16],ctxt_real_BTS2_bundle[15][i%16],true);

        }
    }else if(ctxt_relu_bundle.size() == 4 && ctxt_relu_bundle[0].size() == 32){
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[0][i],ctxt_real_BTS2_bundle[0][i],true);
        }
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[1][i],ctxt_real_BTS2_bundle[1][i],true);
        }
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[2][i],ctxt_real_BTS2_bundle[2][i],true);
        }
        #pragma omp parallel for num_threads(32)
        for(int i = 0 ; i < 32 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[3][i],ctxt_real_BTS2_bundle[3][i],true);
        }
    }else if(ctxt_relu_bundle.size() == 8 && ctxt_relu_bundle[0].size()==8){
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[i/8][i%8],ctxt_real_BTS2_bundle[i/8][i%8],true);
        }
        #pragma omp parallel for num_threads(24)
        for(int i = 0 ; i < 24 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[i/8 + 5][i%8],ctxt_real_BTS2_bundle[i/8 + 5][i%8],true);
        }
    }else{
        #pragma omp parallel for num_threads(40)
        for(int i = 0 ; i < 40 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[0][i],ctxt_real_BTS2_bundle[0][i],true);
        }
        #pragma omp parallel for num_threads(24)
        for(int i = 0 ; i < 24 ; i++){
            eval.bootstrap(ctxt_temp2_bundle[0][i+40],ctxt_real_BTS2_bundle[0][i+40],true);
        }
    }

    
    ctxt_temp2_bundle.clear();
    ctxt_temp2_bundle.shrink_to_fit();

    //std::cout << "e" << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_sign_bundle(ctxt_relu_bundle.size() , std::vector<HEaaN::Ciphertext>(ctxt_relu_bundle[0].size(),ctxt_temp));
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < ctxt_sign_bundle.size() ; ++i){
        for(int j = 0 ; j < ctxt_sign_bundle[0].size() ; ++j){
            evalOddPolynomial(context,eval,ctxt_real_BTS2_bundle[i][j],ctxt_sign_bundle[i][j],polynomial_3,4,3);
        }
    }
    ctxt_real_BTS2_bundle.clear();
    ctxt_real_BTS2_bundle.shrink_to_fit();

    //std::cout << "f" << std::endl;
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < ctxt_relu_bundle.size() ; ++i){
        for(int j = 0 ; j < ctxt_relu_bundle[0].size() ; ++j){
            eval.mult(ctxt_real_BTS_bundle[i][j] , 0.5 , ctxt_real_BTS_bundle[i][j]);
            eval.mult(ctxt_real_BTS_bundle[i][j],ctxt_sign_bundle[i][j],ctxt_relu_bundle[i][j]);
            eval.add(ctxt_real_BTS_bundle[i][j],ctxt_relu_bundle[i][j],ctxt_relu_bundle[i][j]);
        }
    }
    ctxt_sign_bundle.clear();
    ctxt_sign_bundle.shrink_to_fit();
    ctxt_real_BTS_bundle.clear();
    ctxt_real_BTS_bundle.shrink_to_fit();
    levelDownBundle(context,pack,eval,ctxt_relu_bundle,5);

}


    //Aproximated ReLU function.
void ApproxReLU(HEaaN::Context context, HEaaN::HomEvaluator eval, HEaaN::Ciphertext& ctxt, HEaaN::Ciphertext& ctxt_relu) {

    // std::cout << "Input ciphertext - level " << ctxt.getLevel()
    //     << std::endl
    //     << std::endl;

    HEaaN::Ciphertext ctxt_temp(context);
    eval.conjugate(ctxt, ctxt_temp);
    eval.add(ctxt_temp, ctxt, ctxt_temp);
    eval.mult(ctxt_temp, 0.5, ctxt_temp);

    HEaaN::Ciphertext ctxt_real_BTS(context);
    std::cout << "Imaginary Removal Bootstrapping ... " << std::endl;

    eval.bootstrap(ctxt_temp, ctxt_real_BTS, true);

    /*std::cout << "Result ciphertext after imaginary removal bootstrapping - level "
        << ctxt_real_BTS.getLevel() << std::endl
        << std::endl;*/

    std::vector<double> polynomial_1 = {
    1.34595769293910e-33, 2.45589415425004e1, 4.85095667238242e-32, -6.69660449716894e2,
    -2.44541235853840e-30, 6.67299848301339e3, 1.86874811944640e-29, -3.06036656163898e4,
    -5.76227817577242e-29, 7.31884032987787e4, 8.53680673009259e-29, -9.44433217050084e4,
    -6.02701474694667e-29, 6.23254094212546e4, 1.62342843661940e-29, -1.64946744117805e4
    };

    /*
    // product 1/2 to coeff of p_2 because we need to calculate 1/2(x+bar(x)) to do BTS.
    for(int  i = 0 ; i < polynomial_1.size() ; ++i){
        polynomial_1[i] = polynomial_1[i]* 0.5;
    }
    */

    std::vector<double> polynomial_2 = {
    1.53261588585630e-47, 9.35625636035439, -3.68972123048249e-46, -5.91638963933626e1,
    1.74254399703303e-45 , 1.48860930626448e2, -3.20672110002213e-45, -1.75812874878582e2,
    2.79115738948645e-45, 1.09111299685955e2, -1.22590309306100e-45, -3.66768839978755e1,
    2.62189142557962e-46, 6.31846290311294, -2.16662326421275e-47, -4.37113415082177e-01
    };

    /*
        // product 1/2 to coeff of p_2 because we need to calculate 1/2(x+bar(x)) to do BTS.
        for(int  i = 0 ; i < polynomial_2.size() ; ++i){
            polynomial_2[i] = polynomial_2[i]*0.5;
        }
    */
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


    HEaaN::Ciphertext ctxt_temp1(context);

    /*std::cout << "1st polynomial evaluation ... " << std::endl << std::endl;*/
    evalOddPolynomial(context, eval, ctxt_real_BTS, ctxt_temp1, polynomial_1, 4, 2);
    /*std::cout << "done" << std::endl << std::endl;

    std::cout << "Ciphertext after evaluating 1st polynomial - level " << ctxt_temp1.getLevel()
        << std::endl
        << std::endl;

    std::cout << "2nd polynomial evaluation ... " << std::endl << std::endl;*/

    HEaaN::Ciphertext ctxt_temp2(context);
    evalOddPolynomial(context, eval, ctxt_temp1, ctxt_temp2, polynomial_2, 2, 3);

    /*std::cout << "done" << std::endl << std::endl;

    std::cout << "Ciphertext after evaluating 2nd polynomial - level " << ctxt_temp2.getLevel()
        << std::endl
        << std::endl;*/

    /*std::cout << "Bootstrapping ... " << std::endl;*/
    HEaaN::Ciphertext ctxt_BTS(context);
    eval.bootstrap(ctxt_temp2, ctxt_BTS, true);

    /*std::cout << "Result ciphertext after bootstrapping - level "
        << ctxt_BTS.getLevel() << std::endl
        << std::endl;*/

    //ctxt_out3 = eval.sign(ctxt)

  /*  std::cout << "3rd polynomial evaluation ... " << std::endl << std::endl;*/

    HEaaN::Ciphertext ctxt_sign(context);
    evalOddPolynomial(context, eval, ctxt_BTS, ctxt_sign, polynomial_3, 4, 3);

  /*  std::cout << "Ciphertext after evaluating 3rd polynomial - level " << ctxt_sign.getLevel()
        << std::endl
        << std::endl;*/

    //ReLU(x)= 0.5(x+x*sign(x)) = 0.5x + x * (0.5 g3 ^ g2 ^g1)(x), x=ctxt_real_BTS;
    //ctxt_sign = (0.5 g3 ^ g2 ^g1)(ctxt_real_BTS);
    
    HEaaN::Ciphertext ctxt_half(context);
    eval.mult(ctxt_real_BTS, 0.5, ctxt_half);
    eval.mult(ctxt_real_BTS, ctxt_sign, ctxt_relu);
    eval.add(ctxt_half, ctxt_relu, ctxt_relu);
    
    
    //eval.mult(ctxt_real_BTS, 0.5, ctxt_real_BTS);
    //eval.mult(ctxt_real_BTS, ctxt_sign, ctxt_relu);
    //eval.add(ctxt_real_BTS, ctxt_relu, ctxt_relu);

    // std::cout << "Evaluating Apporximate ReLU done; Ciphertext after evaluating approximate ReLU - level " << ctxt_relu.getLevel()
    //     << std::endl
    //     << std::endl;
}


void ApproxReLU_bundle80(HEaaN::Context context, HEaaN::KeyPack pack,HEaaN::HomEvaluator eval, 
    std::vector<std::vector<HEaaN::Ciphertext>>& ctxt_bundle, std::vector<std::vector<HEaaN::Ciphertext>>& ctxt_relu_bundle){
    
    int n1 = ctxt_bundle.size();
    int n2 = ctxt_bundle[0].size();
    
    Ciphertext ctxt_init(context); //for initializing
    vector<vector<Ciphertext>> ctxt_temp_bundle(n1 , vector<Ciphertext>(n2 ,ctxt_init));
   
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < n1 ; i++){
        for(int j = 0 ; j < n2 ; j++){
            eval.conjugate(ctxt_bundle[i][j],ctxt_temp_bundle[i][j]);
            eval.add(ctxt_temp_bundle[i][j],ctxt_bundle[i][j],ctxt_temp_bundle[i][j]);
            eval.mult(ctxt_temp_bundle[i][j],0.5,ctxt_temp_bundle[i][j]);
        }
    }

    //std::cout << "Imaginary BTS..." << std::endl;
    vector<vector<Ciphertext>> ctxt_real_BTS_bundle(n1 , vector<Ciphertext>(n2 ,ctxt_init));
    
    if(n1 == 16 && n2 == 16){
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; i++){
            eval.bootstrap(ctxt_temp_bundle[i/16][i%16],ctxt_real_BTS_bundle[i/16][i%16],true);
        }
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; i++){
            eval.bootstrap(ctxt_temp_bundle[(i/16)+5][i%16],ctxt_real_BTS_bundle[i/16+5][i%16],true);
        }
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; i++){
            eval.bootstrap(ctxt_temp_bundle[(i/16)+10][i%16],ctxt_real_BTS_bundle[i/16+10][i%16],true);
        }
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 16 ; i++){
            #pragma omp parallel num_threads(5)
            {
                eval.bootstrap(ctxt_temp_bundle[15][i%16],ctxt_real_BTS_bundle[15][i%16],true);
            }
        }
        
    }else if(n1 == 4 && n2 == 32){
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; ++i){
            {
                eval.bootstrap(ctxt_temp_bundle[i/20][i%20],ctxt_real_BTS_bundle[i/20][i%20],true);
            }
        }
        #pragma omp parallel for num_threads(48)
        for(int i = 0 ; i < 48 ; ++i){
            {
                eval.bootstrap(ctxt_temp_bundle[i/12][20+(i%12)],ctxt_real_BTS_bundle[i/12][20+(i%12)],true);
            }
        }
        
    }else{
        #pragma omp parallel for num_threads(64)
        for(int i = 0 ; i < 64 ; i++){
            eval.bootstrap(ctxt_temp_bundle[0][i],ctxt_real_BTS_bundle[0][i],true);
        }
    }

    

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
    
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < n1 ; ++i){
        for(int j = 0 ; j < n2 ; ++j){
            evalOddPolynomial(context,eval,ctxt_real_BTS_bundle[i][j],ctxt_temp_bundle[i][j],polynomial_1,4,2);
        }
    }
    
    vector<vector<Ciphertext>> ctxt_temp1_bundle(n1, vector<Ciphertext>(n2, ctxt_init));
    
    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < n1 ; ++i){
        for(int j = 0 ; j < n2 ; ++j){
            evalOddPolynomial(context,eval,ctxt_temp_bundle[i][j],ctxt_temp1_bundle[i][j],polynomial_2,2,3);
        }
    }
    
    //BTS...
   
    if(n1 == 16 && n2 == 16){
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; i++){
            eval.bootstrap(ctxt_temp1_bundle[i/16][i%16],ctxt_temp_bundle[i/16][i%16],true);
        }
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; i++){
            eval.bootstrap(ctxt_temp1_bundle[(i/16)+5][i%16],ctxt_temp_bundle[i/16+5][i%16],true);
        }
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; i++){
            eval.bootstrap(ctxt_temp1_bundle[(i/16)+10][i%16],ctxt_temp_bundle[i/16+10][i%16],true);
        }
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 16 ; i++){
            #pragma omp parallel num_threads(5)
            {
                eval.bootstrap(ctxt_temp1_bundle[15][i%16],ctxt_temp_bundle[15][i%16],true);
            }
        }
    }else if(n1 == 4 && n2 == 32){
        #pragma omp parallel for num_threads(80)
        for(int i = 0 ; i < 80 ; ++i){
            {
                eval.bootstrap(ctxt_temp1_bundle[i/20][i%20],ctxt_temp_bundle[i/20][i%20],true);
            }
        }
        #pragma omp parallel for num_threads(48)
        for(int i = 0 ; i < 48 ; ++i){
            //#pragma omp parallel num_threads(5)
            {
            eval.bootstrap(ctxt_temp1_bundle[i/12][20+(i%12)],ctxt_temp_bundle[i/12][20+(i%12)],true);
            }
        }
    }else{
        #pragma omp parallel for num_threads(64)
        for(int i = 0 ; i < 64 ; i++){
            eval.bootstrap(ctxt_temp1_bundle[0][i],ctxt_temp_bundle[0][i],true);
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < n1 ; ++i){
        for(int j = 0 ; j < n2 ; ++j){
            evalOddPolynomial(context, eval, ctxt_temp_bundle[i][j], ctxt_temp1_bundle[i][j], polynomial_3,4,3);
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0 ; i < n1 ; ++i){
        for(int j = 0 ; j < n2 ; ++j){
            eval.mult(ctxt_real_BTS_bundle[i][j] , 0.5 , ctxt_temp_bundle[i][j]);
            eval.mult(ctxt_real_BTS_bundle[i][j], ctxt_temp1_bundle[i][j], ctxt_relu_bundle[i][j]);
            eval.add(ctxt_temp_bundle[i][j], ctxt_relu_bundle[i][j], ctxt_relu_bundle[i][j]);
        }
    }
    ctxt_temp_bundle.clear();
    ctxt_temp_bundle.shrink_to_fit();
    ctxt_temp1_bundle.clear();
    ctxt_temp1_bundle.shrink_to_fit();
    ctxt_real_BTS_bundle.clear();
    ctxt_real_BTS_bundle.shrink_to_fit();
    
    levelDownBundle(context,pack,eval,ctxt_relu_bundle,5);

}
