#include "kernelEncode.hpp"
#include <math.h>
#include "leveldown.hpp"

namespace {
using namespace HEaaN;
using namespace std;
}



std::vector<std::vector<HEaaN::Ciphertext>> RB(HEaaN::Context context, HEaaN::KeyPack pack,
HEaaN::HomEvaluator eval, int DSB_count, std::vector<std::vector<HEaaN::Ciphertext>>& ctxt_bundle, 
std::vector<std::vector<std::vector<HEaaN::Plaintext>>>& kernel_bundle, 
std::vector<std::vector<std::vector<HEaaN::Plaintext>>>& kernel_bundle2,
vector<Plaintext>& BN1_add,
vector<Plaintext>& BN2_add) {
    ///////////////////////// SetUp ////////////////////////////////
    std::cout << "RB start" << "\n";

    levelDownBundle(context, pack, eval, ctxt_bundle, 5);

    // int num_ctxt;
    // num_ctxt = ctxt_bundle.size();

    // int num_kernel_bundle1;
    // num_kernel_bundle1 = kernel_bundle.size();

    // int num_kernel_bundle2;
    // num_kernel_bundle2 = kernel_bundle2.size();


    ///////////////////////// Main flow /////////////////////////////////////////
    std::cout << "First Conv-(main flow) ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_conv_out_bundle;
    for (int i = 0; i < 16/pow(4, DSB_count); ++i) { 
        //std::cout << "i = " << i << std::endl;
        std::vector<HEaaN::Ciphertext> ctxt_conv_out_cache;
        ctxt_conv_out_cache = Conv(context, pack, eval, 32, 1, 1, 16*pow(2, DSB_count), 16*pow(2, DSB_count), ctxt_bundle[i], kernel_bundle);
        ctxt_conv_out_bundle.push_back(ctxt_conv_out_cache);
    }
    std::cout << "DONE!" << "\n";

    cout << "Adding BN-(main flow) ..." << endl;
    addBNsummands(context, eval,ctxt_conv_out_bundle, BN1_add, 16/pow(4, DSB_count), 16*pow(2, DSB_count));
    cout << "DONE!" << "\n";


    



    // AppReLU
    std::cout << "AppReLU-(main flow) ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_relu_out_bundle;
    for (int i = 0; i < 16/pow(4, DSB_count); ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_relu_out_allch_bundle;
        for (int ch = 0; ch < 16*pow(2, DSB_count); ++ch) {
            std::cout << ch << "\n";
            HEaaN::Ciphertext ctxt_relu_out(context);
            ApproxReLU(context, eval, ctxt_conv_out_bundle[i][ch], ctxt_relu_out);
            eval.levelDown(ctxt_relu_out, 5, ctxt_relu_out);
            ctxt_relu_out_allch_bundle.push_back(ctxt_relu_out);
        }
        ctxt_relu_out_bundle.push_back(ctxt_relu_out_allch_bundle);
    }
    std::cout << "DONE!" << "\n";
    ctxt_conv_out_bundle.clear();
    ctxt_conv_out_bundle.shrink_to_fit();

    // Second convolution
    std::cout << "Second Conv-(main flow) ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_conv_out2_bundle;
    for (int i = 0; i < 16/pow(4, DSB_count); ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_conv_out2_allch_bundle;
        ctxt_conv_out2_allch_bundle = Conv(context, pack, eval, 32, 1, 1, 16*pow(2, DSB_count), 16*pow(2, DSB_count), ctxt_relu_out_bundle[i], kernel_bundle2);
        ctxt_conv_out2_bundle.push_back(ctxt_conv_out2_allch_bundle);
    }
    ctxt_relu_out_bundle.clear();
    ctxt_relu_out_bundle.shrink_to_fit();
    std::cout << "DONE!" << "\n";

    cout << "Adding BN-(main flow) ..." << endl;
    addBNsummands(context, eval,ctxt_conv_out2_bundle, BN2_add, 16/pow(4, DSB_count), 16*pow(2, DSB_count));
    cout << "DONE!" << "\n";


    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    std::cout << "Main flow + Residual flow ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_residual_added;
    for (int i = 0; i < 16/pow(4, DSB_count); ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_residual_added_allch_bundle;
        for (int ch = 0; ch < 16*pow(2, DSB_count); ++ch) {
            HEaaN::Ciphertext ctxt_residual_added_cache(context);
            eval.add(ctxt_conv_out2_bundle[i][ch], ctxt_bundle[i][ch], ctxt_residual_added_cache);
            ctxt_residual_added_allch_bundle.push_back(ctxt_residual_added_cache);
        }
        ctxt_residual_added.push_back(ctxt_residual_added_allch_bundle);
    }
    std::cout << "DONE!" << "\n";

    // Last AppReLU
    std::cout << "Last AppReLU ..." << std::endl;
    std::vector<std::vector<HEaaN::Ciphertext>> ctxt_RB_out;
    for (int i = 0; i < 16/pow(4, DSB_count); ++i) {
        std::vector<HEaaN::Ciphertext> ctxt_RB_out_allch_bundle;
        for (int ch = 0; ch < 16*pow(2, DSB_count); ++ch) {
            HEaaN::Ciphertext ctxt_RB_out_cache(context);
            ApproxReLU(context, eval, ctxt_residual_added[i][ch], ctxt_RB_out_cache);
            ctxt_RB_out_allch_bundle.push_back(ctxt_RB_out_cache);
        }
        ctxt_RB_out.push_back(ctxt_RB_out_allch_bundle);
    }
    ctxt_residual_added.clear();
    ctxt_residual_added.shrink_to_fit();
    std::cout << "DONE!" << "\n";


    return ctxt_RB_out;
}
