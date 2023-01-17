#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

#include "HEaaN/heaan.hpp" // 필요한가?
#include "examples.hpp" // 필요한가? / ㅇㅇ
#include "Conv.hpp"
#include "Conv_parallel.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "DSB+BN.hpp"
#include "RB+BN.hpp"
//#include "convtools.hpp"
#include "kernelEncode.hpp"
#include "imageEncode.hpp"
#include "AvgpoolFC64.hpp"

namespace {
    using namespace HEaaN;
    using namespace std;
}

int main() {
    // SetUp
    HEaaNTimer timer(false);
    ParameterPreset preset = ParameterPreset::FGb;
    Context context = makeContext(preset);
    if (!isBootstrappableParameter(context)) {
        // cout << "Bootstrap is not available for parameter "
        //     << presetNamer(preset) << endl;
        return -1;
    }
    // cout << "Parameter : " << presetNamer(preset) << endl
    //     << endl;
    const auto log_slots = getLogFullSlots(context);

    SecretKey sk(context);
    KeyPack pack(context);

    KeyGenerator keygen(context, sk, pack);

    cout << "Generate encryption key ... " << endl;
    keygen.genEncryptionKey();
    cout << "done" << endl << endl;

    makeBootstrappable(context);

    cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << endl;
    keygen.genCommonKeys();
    cout << "done" << endl << endl;

    Encryptor enc(context);
    Decryptor dec(context);

    cout << "Generate HomEvaluator (including pre-computing constants for "
        "bootstrapping) ..."
        << endl;
    timer.start("* ");
    HomEvaluator eval(context, pack);
    timer.end();

    EnDecoder ecd(context);



    ///////////// Message ///////////////////


    cout << "Image Loading ..." << "\n";
    vector<vector<Ciphertext>> imageVec;

    for (int i = 1; i < 17; ++i) { // 313

        string str = "/app/HEAAN-ResNet/image/image_" + to_string(i) + string(".txt");
        vector<double> temp;
        txtreader(temp, str);
        vector<Ciphertext> out;
        imageCompiler(context, pack, enc, 5, temp, out);
        imageVec.push_back(out);

    }

    cout << "DONE, test for image encode ..." << "\n";

    Message dmsg;
    dec.decrypt(imageVec[0][0], sk, dmsg);
    printMessage(dmsg);

    cout << "DONE" << "\n";

    Ciphertext ctxt_init(context);
    Plaintext ptxt_init(context); // for initializing

    // 1st conv
    vector<double> temp0;
    vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3(16, vector<vector<Plaintext>>(3, vector<Plaintext>(9, ptxt_init)));
    string path0 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block0conv0multiplicands16_3_3_3");
    double cnst = (double)1 / ((double)40);
    Scaletxtreader(temp0, path0, cnst);
    //cout << "conv0 done" <<endl;
    kernel_ptxt(context, temp0, block0conv0multiplicands16_3_3_3, 5, 1, 1, 16, 3, 3, ecd);

    temp0.clear();
    temp0.shrink_to_fit();


    vector<Plaintext> block0conv0summands16;
    vector<double> temp0a;
    string path0a = "/app/HEAAN-ResNet/kernel/summands/" + string("block0conv0summands16");
    Scaletxtreader(temp0a, path0a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp0a[i]);
        block0conv0summands16.push_back(ecd.encode(msg, 4, 0));
    }

    temp0a.clear();
    temp0a.shrink_to_fit();




    // parallel Convolution 1
    cout << "parallel Convolution 1 ..." << endl;
    timer.start(" parallel Convolution 1 ");
    vector<vector<Ciphertext>> ctxt_conv1_out_bundle_par;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        vector<Ciphertext> ctxt_conv1_out_cache;
        ctxt_conv1_out_cache = Conv_parallel(context, pack, eval, 32, 1, 1, 3, 16, imageVec[i], block0conv0multiplicands16_3_3_3);
        ctxt_conv1_out_bundle_par.push_back(ctxt_conv1_out_cache);
    }
    addBNsummands(context, eval, ctxt_conv1_out_bundle_par, block0conv0summands16, 16, 16);
    timer.end();

    imageVec.clear();
    imageVec.shrink_to_fit();
    block0conv0multiplicands16_3_3_3.clear();
    block0conv0multiplicands16_3_3_3.shrink_to_fit();
    block0conv0summands16.clear();
    block0conv0summands16.shrink_to_fit();




    cout << "DONE!... after parallel conv1, result = " << "\n";

    dec.decrypt(ctxt_conv1_out_bundle_par[0][0], sk, dmsg);
    printMessage(dmsg);



    vector<double> temp7;
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path7 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block4conv_onebyone_multiplicands32_16_1_1");
    txtreader(temp7, path7);
    kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 6, 1, 2, 32, 16, 1, ecd);
    temp7.clear();
    temp7.shrink_to_fit();


    vector<Plaintext> block4conv_onebyone_summands32;
    vector<double> temp7a;
    string path7a = "/app/HEAAN-ResNet/kernel/summands/" + string("block4conv_onebyone_summands32");
    Scaletxtreader(temp7a, path7a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp7a[i]);
        block4conv_onebyone_summands32.push_back(ecd.encode(msg, 4, 0));
    }
    temp7a.clear();
    temp7a.shrink_to_fit();





    cout << "parallel block4conv_onebyone ..." << endl;
    cout << "level of ctxt is " << ctxt_conv1_out_bundle_par[0][0].getLevel() << "\n";
    timer.start(" parallel block4conv_onebyone .. ");
    vector<vector<Ciphertext>> ctxt_block4conv_onebyone_out_par;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        vector<Ciphertext> ctxt_residual_out_cache;
        ctxt_residual_out_cache = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_conv1_out_bundle_par[i], block4conv_onebyone_multiplicands32_16_1_1);
        ctxt_block4conv_onebyone_out_par.push_back(ctxt_residual_out_cache);
    }

    block4conv_onebyone_multiplicands32_16_1_1.clear();
    block4conv_onebyone_multiplicands32_16_1_1.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block4conv_onebyone_out_par[0][0].getLevel() << "\n";
    cout << "and decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block4conv_onebyone_out_par[0][0], sk, dmsg);
    printMessage(dmsg);

    // MPP input bundle making
    cout << "parallel block4MPP1 and BN summand ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block4MPP1_in_par(3, vector<vector<Ciphertext>>(32, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block4MPP1_in_par[i][ch][k] = ctxt_block4conv_onebyone_out_par[i + k][ch];
            }
        }
    }
    ctxt_block4conv_onebyone_out_par.clear();
    ctxt_block4conv_onebyone_out_par.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block4MPP1_out_par(4, vector<Ciphertext>(32, ctxt_init));

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            ctxt_block4MPP1_out_par[i][ch] = MPPacking(context, pack, eval, 32, ctxt_block4MPP1_in_par[i][ch]);
        }
    }

    ctxt_block4MPP1_in_par.clear();
    ctxt_block4MPP1_in_par.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block4MPP1_out_par, block4conv_onebyone_summands32, 4, 32);
    timer.end();

    block4conv_onebyone_summands32.clear();
    block4conv_onebyone_summands32.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block4MPP1_out_par[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block4MPP1_out_par[0][0], sk, dmsg);
    printMessage(dmsg);


 
    return 0;
}
