#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

#include "HEaaN/heaan.hpp" 
#include "examples.hpp" 
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
        imageCompiler(context, pack, enc, 6, temp, out);
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
    double cnst = (double)(1.0/40.0);
    Scaletxtreader(temp0, path0, cnst);
    //cout << "conv0 done" <<endl;
    kernel_ptxt(context, temp0, block0conv0multiplicands16_3_3_3, 6, 1, 1, 16, 3, 3, ecd);

    temp0.clear();
    temp0.shrink_to_fit();


    vector<Plaintext> block0conv0summands16;
    vector<double> temp0a;
    string path0a = "/app/HEAAN-ResNet/kernel/summands/" + string("block0conv0summands16");
    Scaletxtreader(temp0a, path0a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp0a[i]);
        block0conv0summands16.push_back(ecd.encode(msg, 5, 0));
    }

    temp0a.clear();
    temp0a.shrink_to_fit();


    cout << "inner parallel Convolution ..." << endl;
    timer.start(" Convolution 1 ");
    vector<vector<Ciphertext>> ctxt_conv1_out_bundle;
    #pragma omp parallel for 
    for (int i = 0; i < 16; ++i) {
        vector<Ciphertext> ctxt_conv1_out_cache;
        ctxt_conv1_out_cache = Conv_parallel(context, pack, eval, 32, 1, 1, 3, 16, imageVec[i], block0conv0multiplicands16_3_3_3);
        ctxt_conv1_out_bundle.push_back(ctxt_conv1_out_cache);
    }
    addBNsummands(context, eval, ctxt_conv1_out_bundle, block0conv0summands16, 16, 16);
    timer.end();

    cout << "DONE!... after conv1, result = " << "\n";

    dec.decrypt(ctxt_conv1_out_bundle[0][0], sk, dmsg);
    printMessage(dmsg);


    
    // parallel Convolution 1
    cout << "outer parallel Convolution 1 ..." << endl;
    timer.start(" Convolution 1 ");
    vector<vector<Ciphertext>> ctxt_conv1_out_bundle_par(16, vector<Ciphertext>(16, ctxt_init));

    #pragma omp parallel for num_threads(40)
    for (int i = 0; i < 8; ++i) {
        #pragma omp parallel num_threads(5)
        {
            ctxt_conv1_out_bundle_par[i] = Conv(context, pack, eval, 32, 1, 1, 3, 16, imageVec[i], block0conv0multiplicands16_3_3_3);
        }    
    }

    #pragma omp parallel for num_threads(40)
    for (int i = 8; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
            ctxt_conv1_out_bundle_par[i] = Conv(context, pack, eval, 32, 1, 1, 3, 16, imageVec[i], block0conv0multiplicands16_3_3_3);
        }    
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







    return 0;

}
