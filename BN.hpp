#pragma once

#include <iostream>
#include "heaan.hpp"
#include <cmath>

// ctxt의 data에 대해서 확인해야 함.
void BN(const double mu , const double std , const double gamma, const double beta ,
        HEaaN::Ciphertext ctxt , HEaaN::HomEvaluator eval, HEaaN::Ciphertext context){
    double epsilon = 0.0001;
    //pre-calculated 
    double divisor = sqrt(pow(std,2)+epsilon);
    double gamma_hat = gamma / divisor;
    double beta_hat = beta - (gamma*mu)/divisor;
    HEaaN::Ciphertext ctxt_out(context);
    eval.mult(ctxt,gamma_hat,ctxt_out);
    eval.add(ctxt_out,beta_hat,ctxt);
}
