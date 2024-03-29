// #include "examples.hpp"
#include <iostream>
#include "HEaaN/heaan.hpp"
#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include <omp.h>

using namespace HEaaN;

    
std::vector<std::vector<Ciphertext>> DSB(Context context, KeyPack pack,
    HomEvaluator eval, int DSB_count, std::vector<std::vector<Ciphertext>> ctxt_bundle, 
    std::vector<std::vector<std::vector<Plaintext>>> kernel_bundle, 
    std::vector<std::vector<std::vector<Plaintext>>> kernel_bundle2, 
    std::vector<std::vector<std::vector<Plaintext>>> kernel_residual_bundle) {
    ///////////////////////// SetUp ////////////////////////////////
    std::cout << "DSB start" << "\n";
    
    HEaaNTimer timer(false); 
    // int num_ctxt;
    // num_ctxt = ctxt_bundle.size();

    // int num_kernel_bundle1;
    // num_kernel_bundle1 = kernel_bundle.size();

    // int num_kernel_bundle2;
    // num_kernel_bundle2 = kernel_bundle2.size();
    
    //HEaaN::Ciphertext ctxt(context); 
    //std::vector<HEaaN::Ciphertext> vector1(32,ctxt); 
    //std::vector<std::vector<HEaaN::Ciphertext>> ctxt_conv_out_bundle; 


    
    ///////////////////////// Main flow /////////////////////////////////////////
    std::cout << "First Conv-(main flow) ..." << std::endl;
    HEaaN::Ciphertext ctxt(context);
    std::vector<HEaaN::Ciphertext> vector1(32,ctxt); 
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_conv_out_bundle;
    //#pragma omp parallel
    //#pragma omp for
    
    //for (int i = 0; i < 16; ++i) { 
    //    std::vector<Ciphertext> ctxt_conv_out_cache;
        //printf("thread %d : %d (in parallel)\n", omp_get_thread_num(), i);
    //    ctxt_conv_out_cache = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_bundle[i], kernel_bundle);
        //printf("thread is done %d : %d (in parallel)\n", omp_get_thread_num(), i);
        //ctxt_conv_out_bundle.push_back(ctxt_conv_out_cache);
    //    ctxt_conv_out_bundle.push_back(ctxt_conv_out_cache); 

    //}
    //std::cout << "DONE!" << "\n";

    

    
    
    /*
    // MPP input bundle making
    //std::vector<std::vector<std::vector<Ciphertext>>> ctxt_MPP_in;
    timer.start("MPP input");
    std::cout << "MPP-(main flow, First Conv) ..." << std::endl;
    HEaaN::Ciphertext ctxt2(context); 
    std::vector<HEaaN::Ciphertext> vector2(32,ctxt2); 
    std::vector<std::vector<HEaaN::Ciphertext>> vector3(32,vector2); 
    std::vector<std::vector<std::vector<Ciphertext>>> ctxt_MPP_in(4,vector3);
    //std::vector<std::vector<HEaaN::Ciphertext>> ctxt_MPP_in_allch_bundle(4,vector2);
    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i < 4; ++i) {
        std::vector<std::vector<Ciphertext>> ctxt_MPP_in_allch_bundle;
        //std::vector<Ciphertext> ctxt_MPP_in_cache(32,ctxt3); 
        for (int ch = 0; ch < 32; ++ch) {
            std::vector<Ciphertext> ctxt_MPP_in_cache;
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+0][ch]);
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+1][ch]);
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+2][ch]);
            ctxt_MPP_in_cache.push_back(ctxt_conv_out_bundle[i+3][ch]);
            ctxt_MPP_in_allch_bundle.push_back(ctxt_MPP_in_cache);
        }
        ctxt_MPP_in[i] = ctxt_MPP_in_allch_bundle;
    }
    timer.end();
   
    

    /*
    // MPP
    //HEaaN::Ciphertext ctxt3(context); // garbage value,추가
    //std::vector<HEaaN::Ciphertext> vector2(32,ctxt3); //추가
    std::vector<std::vector<Ciphertext>> ctxt_MPP_out_bundle;
    //std::vector<HEaaN::Ciphertext> ctxt_MPP_out(32,ctxt3) //추가
    //#pragma omp parallel
    //#pragma omp for
    for (int i = 0; i < 4; ++i) {
        std::vector<Ciphertext> ctxt_MPP_out;
        //#pragma omp parallel
        //#pragma omp for
        for (int ch = 0; ch < 32; ++ch) {
            Ciphertext ctxt_MPP_out_cache(context);
            ctxt_MPP_out_cache = MPPacking(context, pack, eval, 32, ctxt_MPP_in[i][ch]);
            //ctxt_MPP_out.push_back(ctxt_MPP_out_cache);
            ctxt_MPP_out.push_back(ctxt_MPP_out_cache);
            //printf("thread %d : %d (in parallel)\n", omp_get_thread_num(), i);

        }
        //ctxt_MPP_out_bundle.push_back(ctxt_MPP_out);
        ctxt_MPP_out_bundle.push_back(ctxt_MPP_out);
    }
    std::cout << "DONE!" << "\n";
    // ctxt_MPP_out_bundle

    
    // /////////////// Decryption ////////////////
    // Message dmsg0;
    // std::cout << "Decrypt ... ";
    // dec.decrypt(ctxt_MPP_out, sk, dmsg0);
    // std::cout << "done" << std::endl;
    // printMessage(dmsg0);
    // ////////////////////////////////////////

    // AppReLU
    std::cout << "AppReLU-(main flow) ..." << std::endl;
    std::vector<std::vector<Ciphertext>> ctxt_relu_out_bundle;
    //#pragma omp parallel 
    //#pragma omp for 
    for (int i = 0; i < 4; ++i) {
        //HEaaN::Ciphertext ctxt3(context); 
        std::vector<Ciphertext> ctxt_relu_out_allch_bundle;
        //#pragma omp parallel
        //#pragma omp for 
        for (int ch = 0; ch < 32; ++ch) {
            std::cout << "(i = " << i << ", " << "ch = " << ch << ")" << "\n";
            Ciphertext ctxt_relu_out(context);
            ApproxReLU(context, eval, ctxt_MPP_out_bundle[i][ch], ctxt_relu_out);
            ctxt_relu_out_allch_bundle.push_back(ctxt_relu_out);
        }
        ctxt_relu_out_bundle.push_back(ctxt_relu_out_allch_bundle);
    }
    std::cout << "DONE!" << "\n";

    
    // Second convolution
    std::cout << "Second Conv-(main flow) ..." << std::endl;
    std::vector<std::vector<Ciphertext>> ctxt_conv_out2_bundle;
    for (int i = 0; i < 4; ++i) {
        std::vector<Ciphertext> ctxt_conv_out2_allch_bundle;
        ctxt_conv_out2_allch_bundle = Conv(context, pack, eval, 32, 1, 1, 32, 32, ctxt_relu_out_bundle[i], kernel_bundle2);
        ctxt_conv_out2_bundle.push_back(ctxt_conv_out2_allch_bundle);
    }
    std::cout << "DONE!" << "\n";

    ///////////////////// Residual flow ////////////////////////////
    // Convolution
    std::cout << "Residual Conv-(residual flow) ..." << std::endl;
    std::vector<std::vector<Ciphertext>> ctxt_residual_out_bundle;
    for (int i = 0; i < 16; ++i) { 
        std::vector<Ciphertext> ctxt_residual_out_cache;
        ctxt_residual_out_cache = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_bundle[i], kernel_residual_bundle);
        ctxt_residual_out_bundle.push_back(ctxt_residual_out_cache);
    }
    std::cout << "DONE!" << "\n";
    
    // MPP input bundle making
    std::cout << "MPP-(residual flow) ..." << std::endl;
    std::vector<std::vector<std::vector<Ciphertext>>> ctxt_MPP_in2;
    for (int i = 0; i < 4; ++i) {
        std::vector<std::vector<Ciphertext>> ctxt_MPP_in_allch_bundle2;
        for (int ch = 0; ch < 32; ++ch) {
            std::vector<Ciphertext> ctxt_MPP_in_cache2;
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+0][ch]);
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+1][ch]);
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+2][ch]);
            ctxt_MPP_in_cache2.push_back(ctxt_residual_out_bundle[i+3][ch]);
            ctxt_MPP_in_allch_bundle2.push_back(ctxt_MPP_in_cache2);
        }
        ctxt_MPP_in2.push_back(ctxt_MPP_in_allch_bundle2);
    }
    // MPP
    std::vector<std::vector<Ciphertext>> ctxt_MPP_out_bundle2;
    for (int i = 0; i < 4; ++i) {
        std::vector<Ciphertext> ctxt_MPP_out2;
        for (int ch = 0; ch < 32; ++ch) {
            Ciphertext ctxt_MPP_out_cache2(context);
            ctxt_MPP_out_cache2 = MPPacking(context, pack, eval, 32, ctxt_MPP_in2[i][ch]);
            ctxt_MPP_out2.push_back(ctxt_MPP_out_cache2);
        }
        ctxt_MPP_out_bundle2.push_back(ctxt_MPP_out2);
    }
    std::cout << "DONE!" << "\n";


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    std::cout << "Main flow + Residual flow ..." << std::endl;
    std::vector<std::vector<Ciphertext>> ctxt_residual_added;
    for (int i = 0; i < 4; ++i) {
        std::vector<Ciphertext> ctxt_residual_added_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            Ciphertext ctxt_residual_added_cache(context);
            eval.add(ctxt_conv_out2_bundle[i][ch], ctxt_MPP_out_bundle2[i][ch], ctxt_residual_added_cache);
            ctxt_residual_added_allch_bundle.push_back(ctxt_residual_added_cache);
        }
        ctxt_residual_added.push_back(ctxt_residual_added_allch_bundle);
    }
    std::cout << "DONE!" << "\n";

    // Last AppReLU
    std::cout << "Last AppReLU ..." << std::endl;
    std::vector<std::vector<Ciphertext>> ctxt_DSB_out;
    for (int i = 0; i < 4; ++i) {
        std::vector<Ciphertext> ctxt_DSB_out_allch_bundle;
        for (int ch = 0; ch < 32; ++ch) {
            std::cout << "(i = " << i << ", " << "ch = " << ch << ")" << "\n";
            Ciphertext ctxt_DSB_out_cache(context);
            ApproxReLU(context, eval, ctxt_residual_added[i][ch], ctxt_DSB_out_cache);
            ctxt_DSB_out_allch_bundle.push_back(ctxt_DSB_out_cache);
        }
        ctxt_DSB_out.push_back(ctxt_DSB_out_allch_bundle);
    }
    std::cout << "DONE!" << "\n";


    */
    //return ctxt_DSB_out;
    return ctxt_conv_out_bundle;
}
