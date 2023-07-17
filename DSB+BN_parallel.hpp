#pragma once
#include "kernelEncode.hpp"
#include <math.h>
#include <omp.h>

namespace {
    using namespace HEaaN;
    using namespace std;
}


vector<vector<Ciphertext>> DSB_parallel(Context context, KeyPack pack,
    HomEvaluator eval, int DSB_count, vector<vector<Ciphertext>>& ctxt_bundle,
    vector<vector<vector<Plaintext>>>& kernel_bundle,
    vector<vector<vector<Plaintext>>>& kernel_bundle2,
    vector<vector<vector<Plaintext>>>& kernel_residual_bundle,
    vector<Plaintext>& BN1_add,
    vector<Plaintext>& BN2_add,
    vector<Plaintext>& BN3_add) {
    ///////////////////////// SetUp ////////////////////////////////
    cout << "DSB start" << "\n";

    const int size1 = 16 / pow(4, DSB_count);
    const int size1a = size1 / 4;
    const int size2 = 16 * pow(2, DSB_count);
    const int size2a = 2 * size2;


    int num_ctxt;
    num_ctxt = ctxt_bundle.size();

    int num_kernel_bundle1;
    num_kernel_bundle1 = kernel_bundle.size();

    int num_kernel_bundle2;
    num_kernel_bundle2 = kernel_bundle2.size();

    Ciphertext ctxt_init(context);

    ///////////////////////// Main flow /////////////////////////////////////////
    cout << "First Conv-(main flow) ..." << endl;
    cout << "level of ctxt is " << ctxt_bundle[0][0].getLevel() << "\n";
    vector<vector<Ciphertext>> ctxt_conv_out_bundle(size, vector<Ciphertext>(size3, ctxt_init));

    #pragma omp parallel for
    for (int i = 0; i < size1; ++i) { 
        ctxt_conv_out_cache[i] = Conv_parallel(context, pack, eval, 32, 1, 2, size2, size2a, ctxt_bundle[i], kernel_bundle);
    }


    cout << "level of ctxt is " << ctxt_conv_out_bundle[0][0].getLevel() << "\n";
    cout << "DONE!" << "\n";
   

    // MPP input bundle making
    cout << "MPP-(main flow, First Conv) ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_MPP_in(size1a, vector<vector<Ciphertext>>(size2a, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < size1a; ++i) {
        for (int ch = 0; ch < size2a; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_MPP_in[i][ch][k] = ctxt_conv_out_bundle[i + k][ch];
            }
        }
    }

    // vector<vector<Ciphertext>>().swap(ctxt_conv_out_bundle);
    ctxt_conv_out_bundle.clear();
    ctxt_conv_out_bundle.shrink_to_fit();

    
    vector<vector<Ciphertext>> ctxt_MPP_out_bundle(size1a, vector<Ciphertext>(size2a, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size1a; ++i) {
        for (int ch = 0; ch < size2a; ++ch) {
            ctxt_MPP_out_bundle[i][ch] = MPPacking(context, pack, eval, 32, ctxt_MPP_in[i][ch]);
        }
    }

    // vector<vector<vector<Ciphertext>>>().swap(ctxt_MPP_in);
    ctxt_MPP_in.clear();
    ctxt_MPP_in.shrink_to_fit();
    cout << "DONE!" << "\n";
    cout << "Adding BN-(main flow) ..." << endl;
    addBNsummands(context, eval, ctxt_MPP_out_bundle, BN1_add, 4 / pow(4, DSB_count), 32 * pow(2, DSB_count));
    // for (int i = 0; i < 4; ++i) {
    //     for (int ch = 0; ch < 32; ++ch) {
    //         // Ciphertext ctxt_BN1_out_bundle_cache(context);
    //         // cout << ctxt_MPP_out_bundle[i][ch].getLogSlots() << "\n";
    //         // cout << BN1_add[ch].getLogSlots() << "\n";
    //         eval.add(ctxt_MPP_out_bundle[i][ch], BN1_add[ch], ctxt_MPP_out_bundle[i][ch]);

    //         // ctxt_MPP_out_bundle[i][ch] = ctxt_BN1_out_bundle_cache;
    //     }
    // }
    cout << "DONE!" << "\n";


    // AppReLU
    cout << "AppReLU-(main flow) ..." << endl;
    vector<vector<Ciphertext>> ctxt_relu_out_bundle(size1a, vector<Ciphertext>(size2a, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size1a; ++i) {
        for (int ch = 0; ch < size2a; ++ch) {
            ApproxReLU(context, eval, ctxt_MPP_out_bundle[i][ch], ctxt_relu_out_bundle[i][ch]);
            eval.levelDown(ctxt_relu_out_bundle[i][ch], 5, ctxt_relu_out_bundle[i][ch]);
        }
    }

    // vector<vector<Ciphertext>>().swap(ctxt_MPP_out_bundle);
    ctxt_MPP_out_bundle.clear();
    ctxt_MPP_out_bundle.shrink_to_fit();
    cout << "DONE!" << "\n";

    // Second convolution
    cout << "Second Conv-(main flow) ..." << endl;
    vector<vector<Ciphertext>> ctxt_conv_out2_bundle(size2a, vector<Ciphertext>(size2a, ctxt_init));

    #pragma omp parallel for
    for (int i = 0; i < size2a; ++i) {
        ctxt_conv_out2_bundle[i] = Conv_parallel(context, pack, eval, 32, 1, 1, size2a, size2a, ctxt_relu_out_bundle[i], kernel_bundle2);
    }

    //cout << "level of ctxt is " << ctxt_conv_out2_bundle[0][0].getLevel() << "\n";
    // vector<vector<Ciphertext>>().swap(ctxt_relu_out_bundle);
    ctxt_relu_out_bundle.clear();
    ctxt_relu_out_bundle.shrink_to_fit();
    cout << "DONE!" << "\n";
    cout << "Adding BN-(main flow) ..." << endl;
    addBNsummands(context, eval, ctxt_conv_out2_bundle, BN2_add, 4, 32);
    cout << "DONE!" << "\n";


    ///////////////////// Residual flow ////////////////////////////
    // Convolution
    cout << "Residual Conv-(residual flow) ..." << endl;
    vector<vector<Ciphertext>> ctxt_residual_out_bundle(size1, vector<Ciphertext>(size2a, ctxt_init));

    #pragma omp parallel for
    for (int i = 0; i < size1; ++i) {
        ctxt_residual_out_bundle[i] = Conv(context, pack, eval, 32, 1, 2, size2, size2a, ctxt_bundle[i], kernel_residual_bundle);
    }

    cout << "DONE!" << "\n";

    // MPP input bundle making
    cout << "MPP-(residual flow) ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_MPP_in2(size1a, vector<vector<Ciphertext>>(size2a, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 4 / pow(4, DSB_count); ++i) {
        for (int ch = 0; ch < 32 * pow(2, DSB_count); ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_MPP_in2[i][ch][k] = ctxt_residual_out_bundle[i + k][ch];
            }
        }
    }
    // vector<vector<Ciphertext>>().swap(ctxt_residual_out_bundle);
    ctxt_residual_out_bundle.clear();
    ctxt_residual_out_bundle.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_MPP_out_bundle2(size1a, vector<Ciphertext>(size2a, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 4 / pow(4, DSB_count); ++i) {
        for (int ch = 0; ch < 32 * pow(2, DSB_count); ++ch) {
            ctxt_MPP_out_cache2[i][ch] = MPPacking(context, pack, eval, 32, ctxt_MPP_in2[i][ch]);
        }
    }
    // vector<vector<vector<Ciphertext>>>().swap(ctxt_MPP_in2);
    ctxt_MPP_in2.clear();
    ctxt_MPP_in2.shrink_to_fit();
    cout << "DONE!" << "\n";

    cout << "Adding BN-(main flow) ..." << endl;
    addBNsummands(context, eval, ctxt_MPP_out_bundle2, BN3_add, 4 / pow(4, DSB_count), 32 * pow(2, DSB_count));
    cout << "DONE!" << "\n";


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Main flow + Residual flow ..." << endl;
    vector<vector<Ciphertext>> ctxt_residual_added(size1a, vector<Ciphertext>(size2a, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 4 / pow(4, DSB_count); ++i) {
        for (int ch = 0; ch < 32 * pow(2, DSB_count); ++ch) {
            eval.add(ctxt_conv_out2_bundle[i][ch], ctxt_MPP_out_bundle2[i][ch], ctxt_residual_added[i][ch]);
        }
    }

    // vector<vector<Ciphertext>>().swap(ctxt_conv_out2_bundle);
    ctxt_conv_out2_bundle.clear();
    ctxt_conv_out2_bundle.shrink_to_fit();
    // vector<vector<Ciphertext>>().swap(ctxt_MPP_out_bundle2);
    ctxt_MPP_out_bundle2.clear();
    ctxt_MPP_out_bundle2.shrink_to_fit();
    cout << "DONE!" << "\n";

    // Last AppReLU
    cout << "Last AppReLU ..." << endl;
    vector<vector<Ciphertext>> ctxt_DSB_out(size1a, vector<Ciphertext>(size2a, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 4 / pow(4, DSB_count); ++i) {
        for (int ch = 0; ch < 32 * pow(2, DSB_count); ++ch) {
            ApproxReLU(context, eval, ctxt_residual_added[i][ch], ctxt_DSB_out[i][ch]);
        }
    }

    // vector<vector<Ciphertext>>().swap(ctxt_residual_added);
    ctxt_residual_added.clear();
    ctxt_residual_added.shrink_to_fit();
    cout << "DONE!" << "\n";


    return ctxt_DSB_out;
}

