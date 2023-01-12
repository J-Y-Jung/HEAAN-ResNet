#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

#include "HEaaN/heaan.hpp" // 필요한가?
// #include "examples.hpp" // 필요한가?
#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "DSB+BN.hpp"
#include "RB+BN.hpp"
#include "convtools.hpp"
#include "kernelEncode.hpp"
#include "imageEncode.hpp"

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
        
        string str = "/app/examples/image/image_" + to_string(i) + string(".txt");
        vector<double> temp;


        txtreader(temp, str);
        vector<Ciphertext> out;
        imageCompiler(context, pack, enc, temp, out);
        imageVec.push_back(out);

    }
    cout << "DONE" << "\n";

    // // string str = "./image/image_" + to_string(313) + ".txt";
    // vector<double> temp;
    // txtreader(temp, str);
    // for (int i = 0; i < 49152; ++i) temp.push_back(0);

    // vector<Ciphertext> out;
    // imageCompiler(context, pack, enc, temp, out);
    // imageVec.push_back(out);



    Plaintext ptxt_init(context); // for initializing


    // 1st conv

    
    vector<double> temp0;
    vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3(16, vector<vector<Plaintext>>(3, vector<Plaintext>(9, ptxt_init)));
    string path0 = "/app/examples/kernel/multiplicands/" + string("block0conv0multiplicands16_3_3_3");
    txtreader(temp0, path0);
    cout << "conv0 done" <<endl;
    kernel_ptxt(context, temp0, block0conv0multiplicands16_3_3_3, 12, 1, 1, 16, 3, 3, ecd);
    
    vector<double> block0conv0summands16;
    string path0a = "/app/examples/kernel/summands/" + string("block0conv0summands16");
    txtreader(block0conv0summands16, path0a);


    // Convolution 1
    cout << "Convolution 1 ..." << endl;
    timer.start(" Convolution 1 ");
    vector<vector<Ciphertext>> ctxt_conv1_out_bundle;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        vector<Ciphertext> ctxt_conv1_out_cache;
        ctxt_conv1_out_cache = Conv(context, pack, eval, 32, 1, 1, 3, 16, imageVec[i], block0conv0multiplicands16_3_3_3);
        ctxt_conv1_out_bundle.push_back(ctxt_conv1_out_cache);
    }
    addBNsummands(context, eval, ctxt_conv1_out_bundle, block0conv0summands16, 16, 16);

    timer.end();
    cout << "DONE!" << "\n";


    //memory delete
    //block0conv0multiplicands16_3_3_3.clear();
    //block0conv0multiplicands16_3_3_3.shrink_to_fit();


    // AppReLU
    cout << "AppReLU ..." << endl;
    timer.start(" AppReLU 1 ");
    vector<vector<Ciphertext>> ctxt_relu1_out_bundle;
    for (int i = 0; i < 16; ++i) {
        vector<Ciphertext> ctxt_relu1_out_allch_bundle;
        for (int ch = 0; ch < 16; ++ch) {
            cout << "(i = " << i << ", " << "ch = " << ch << ")" << "\n";
            Ciphertext ctxt_relu1_out(context);
            ApproxReLU(context, eval, ctxt_conv1_out_bundle[i][ch], ctxt_relu1_out);
            ctxt_relu1_out_allch_bundle.push_back(ctxt_relu1_out);
        }
        ctxt_relu1_out_bundle.push_back(ctxt_relu1_out_allch_bundle);
    }
    timer.end();
    cout << "DONE!" << "\n";


    // Residual Block 1, 2, 3
        // RB 1 - 1
    vector<double> temp1;
    vector<vector<vector<Plaintext>>> block1conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path1 = "/app/examples/kernel/multiplicands/" + string("block1conv0multiplicands16_16_3_3");
    txtreader(temp1, path1);
    kernel_ptxt(context, temp1, block1conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);
    
    vector<double> block1conv0summands16;
    string path1a = "/app/examples/kernel/summands/" + string("block1conv0summands16");
    txtreader(block1conv0summands16, path1a);
    
        // RB 1 - 2
    vector<double> temp2;
    vector<vector<vector<Plaintext>>> block1conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path2 = "/app/examples/kernel/multiplicands/" + string("block1conv1multiplicands16_16_3_3");
    txtreader(temp2, path2);
    kernel_ptxt(context, temp2, block1conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block1conv1summands16;
    string path2a = "/app/examples/kernel/summands/" + string("block1conv1summands16");
    txtreader(block0conv0summands16, path2a);

    // RB 1
    cout << "RB 1 ..." << endl;
    timer.start(" RB 1 ");
    vector<vector<Ciphertext>> ctxt_RB1_out;
    ctxt_RB1_out = RB(context, pack, eval, 0, ctxt_relu1_out_bundle, block1conv0multiplicands16_16_3_3, block1conv1multiplicands16_16_3_3,
    block1conv0summands16, block1conv1summands16);
    timer.end();
    cout << "DONE!" << "\n";

    //memeory delete    
    // ctxt_relu1_out_bundle.claer();
    // ctxt_relu1_out_bundle.shrink_to_fit();
    // block1conv0multiplicands16_16_3_3.clear();
    // block1conv0multiplicands16_16_3_3.shrink_to_fit();
    // block1conv1multiplicands16_16_3_3.clear();
    // block1conv1multiplicands16_16_3_3.shrink_to_fit();
    // block1conv0summands16.clear();
    // block1conv0summands16.shrink_to_fit();
    // block1conv1summands16.clear();
    // block1conv1summands16.shrink_to_fit();




        // RB 2 - 1
    vector<double> temp3;
    vector<vector<vector<Plaintext>>> block2conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path3 = "/app/examples/kernel/multiplicands/" + string("block2conv0multiplicands16_16_3_3");
    txtreader(temp3, path3);
    kernel_ptxt(context, temp3, block2conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block2conv0summands16;
    string path3a = "/app/examples/kernel/summands/" + string("block2conv0summands16");
    txtreader(block2conv0summands16, path3a);
    
    // RB 2 - 2
    vector<double> temp4;
    vector<vector<vector<Plaintext>>> block2conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path4 = "/app/examples/kernel/multiplicands/" + string("block2conv1multiplicands16_16_3_3");
    txtreader(temp4, path4);
    kernel_ptxt(context, temp4, block2conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block2conv1summands16;
    string path4a = "/app/examples/kernel/summands/" + string("block2conv1summands16");
    txtreader(block2conv1summands16, path4a);




    // RB 2
    cout << "RB 2..." << endl;
    timer.start(" RB 2 ");
    vector<vector<Ciphertext>> ctxt_RB2_out;
    ctxt_RB2_out = RB(context, pack, eval, 0, ctxt_RB1_out, block2conv0multiplicands16_16_3_3, block2conv1multiplicands16_16_3_3,
    block2conv0summands16, block2conv1summands16);
    timer.end();
    cout << "DONE!" << "\n";

    //memory delete 
    // ctxt_RB1_out.clear();
    // ctxt_RB1_out.shrink_to_fit();
    // block2conv0multiplicands16_16_3_3.clear();
    // block2conv0multiplicands16_16_3_3.shrink_to_fit();
    // block2conv1multiplicands16_16_3_3.clear();
    // block2conv1multiplicands16_16_3_3.shrink_to_fit();
    // block2conv0summands16.clear();
    // block2conv0summands16.shrink_to_fit();
    // block2conv1summands16.clear();
    // block2conv1summands16.shrink_to_fit();





        // RB 3 - 1
    vector<double> temp5;
    vector<vector<vector<Plaintext>>> block3conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path5 = "/app/examples/kernel/multiplicands/" + string("block3conv0multiplicands16_16_3_3");
    txtreader(temp5, path5);
    kernel_ptxt(context, temp5, block3conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block3conv0summands16;
    string path5a = "/app/examples/kernel/summands/" + string("block3conv0summands16");
    txtreader(block3conv0summands16, path5a);
    
    
    // RB 3 - 2
    vector<double> temp6;
    vector<vector<vector<Plaintext>>> block3conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path6 = "/app/examples/kernel/multiplicands/" + string("block3conv1multiplicands16_16_3_3");
    txtreader(temp6, path6);
    kernel_ptxt(context, temp6, block3conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block3conv1summands16;
    string path6a = "/app/examples/kernel/summands/" + string("block3conv1summands16");
    txtreader(block3conv1summands16, path5a);

    // RB 3
    cout << "RB 3 ..." << endl;
    timer.start(" RB 3 ");
    vector<vector<Ciphertext>> ctxt_RB3_out;
    ctxt_RB3_out = RB(context, pack, eval, 0, ctxt_RB2_out, block3conv0multiplicands16_16_3_3, block3conv1multiplicands16_16_3_3,
    block3conv0summands16, block3conv1summands16);
    timer.end();
    cout << "DONE!" << "\n";

    //memory delete 
    // ctxt_RB2_out.clear();
    // ctxt_RB2_out.shrink_to_fit();
    // block3conv0multiplicands16_16_3_3.clear();
    // block3conv0multiplicands16_16_3_3.shrink_to_fit();
    // block3conv1multiplicands16_16_3_3.clear();
    // block3conv1multiplicands16_16_3_3.shrink_to_fit();
    // block3conv0summands16.clear();
    // block3conv0summands16.shrink_to_fit();
    // block3conv1summands16.clear();
    // block3conv1summands16.shrink_to_fit();




    // Down Sampling (Residual) Block 1
        // DSB 1 - res
    vector<double> temp7;
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path7 = "/app/examples/kernel/multiplicands/" + string("block4conv_onebyone_multiplicands32_16_1_1");
    txtreader(temp7, path7);
    kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 12, 1, 2, 32, 16, 1, ecd);

    
    vector<double> block4conv_onebyone_summands32;
    string path7a = "/app/examples/kernel/summands/" + string("block4conv_onebyone_summands16");
    txtreader(block4conv_onebyone_summands32, path7a);
    
    // DSB 1 - 1
    vector<double> temp8;
    vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path8 = "/app/examples/kernel/multiplicands/" + string("block4conv0multiplicands32_16_3_3");
    txtreader(temp8, path8);
    kernel_ptxt(context, temp8, block4conv0multiplicands32_16_3_3, 12, 1, 2, 32, 16, 3, ecd);


    vector<double> block4conv0summands32;
    string path8a = "/app/examples/kernel/summands/" + string("block4conv0summands32");
    txtreader(block4conv0summands32, path8a);
    
    
    // DSB 1 - 2
    vector<double> temp9;
    vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path9 = "/app/examples/kernel/multiplicands/" + string("block4conv1multiplicands32_32_3_3");
    txtreader(temp9, path9);
    kernel_ptxt(context, temp9, block4conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);


    vector<double> block4conv1summands32;
    string path9a = "/app/examples/kernel/summands/" + string("block4conv1summands32");
    txtreader(block4conv1summands32, path9a);

    // DSB 1
    cout << "DSB 1 ..." << endl;
    timer.start(" DSB 1 ");
    vector<vector<Ciphertext>> ctxt_DSB1_out;
    ctxt_DSB1_out = DSB(context, pack, eval, 0, ctxt_RB3_out, block4conv0multiplicands32_16_3_3, block4conv1multiplicands32_32_3_3, block4conv_onebyone_multiplicands32_16_1_1,
    block4conv0summands32, block4conv1summands32, block4conv_onebyone_summands32);
    cout << "DONE!" << "\n";

    //memory delete
    // ctxt_RB3_out.clear();
    // ctxt_RB3_out.shrink_to_fit();
    // block4conv0multiplicands32_16_3_3.clear();
    // block4conv0multiplicands32_16_3_3.shrink_to_fit();
    // block4conv1multiplicands32_32_3_3.clear();
    // block4conv1multiplicands32_32_3_3.shrink_to_fit();
    // block4conv_onebyone_multiplicands32_16_1_1.clear();
    // block4conv_onebyone_multiplicands32_16_1_1.shrink_to_fit();
    // block4conv0summands32.clear();
    // block4conv0summands32.shrink_to_fit();
    // block4conv1summands32.clear();
    // block4conv1summands32.shrink_to_fit();
    // block4conv_onebyone_summands32.clear();
    // block4conv_onebyone_summands32.shrink_to_fit();



    // Residual Block 4, 5
        // RB 4 - 1
    vector<double> temp10;
    vector<vector<vector<Plaintext>>> block5conv0multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path10 = "/app/examples/kernel/multiplicands/" + string("block5conv0multiplicands32_32_3_3");
    txtreader(temp10, path10);
    kernel_ptxt(context, temp10, block5conv0multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);
    
    
    vector<double> block5conv0summands32;
    string path10a = "/app/examples/kernel/summands/" + string("block5conv0summands32");
    txtreader(block5conv0summands32, path10a);
    
    
    // RB 4 - 2
    vector<double> temp11;
    vector<vector<vector<Plaintext>>> block5conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path11 = "/app/examples/kernel/multiplicands/" + string("block5conv1multiplicands32_32_3_3");
    txtreader(temp11, path11);
    kernel_ptxt(context, temp11, block5conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);
    
    
    vector<double> block5conv1summands32;
    string path11a = "/app/examples/kernel/summands/" + string("block5conv1summands32");
    txtreader(block5conv1summands32, path11a);

    // RB 4
    cout << "RB 4..." << endl;
    timer.start(" RB 4 ");
    vector<vector<Ciphertext>> ctxt_RB4_out;
    ctxt_RB4_out = RB(context, pack, eval, 1, ctxt_DSB1_out, block5conv0multiplicands32_32_3_3, block5conv1multiplicands32_32_3_3,
    block5conv0summands32, block5conv1summands32);
    timer.end();
    cout << "DONE!" << "\n";

    //memory delete 
    // ctxt_DSB1_out.clear();
    // ctxt_DSB1_out.shrink_to_fit();
    // block5conv0multiplicands32_32_3_3.claer();
    // block5conv0multiplicands32_32_3_3.shrink_to_fit();
    // block5conv1multiplicands32_32_3_3.clear();
    // block5conv1multiplicands32_32_3_3.shrink_to_fit();
    // block5conv0summands32.clear();
    // block5conv0summands32.shrink_to_fit();
    // block5conv1summands32.clear();
    // block5conv1summands32.shrink_to_fit();








        // RB 5 - 1
    vector<double> temp12;
    vector<vector<vector<Plaintext>>> block6conv0multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path12 = "/app/examples/kernel/multiplicands/" + string("block6conv0multiplicands32_32_3_3");
    txtreader(temp12, path12);
    kernel_ptxt(context, temp12, block6conv0multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);

    vector<double> block6conv0summands32;
    string path12a = "/app/examples/kernel/summands/" + string("block6conv0summands32");
    txtreader(block6conv0summands32, path12a);
    
    // RB 5 - 2
    vector<double> temp13;
    vector<vector<vector<Plaintext>>> block6conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path13 = "/app/examples/kernel/multiplicands/" + string("block6conv1multiplicands32_32_3_3");
    txtreader(temp13, path13);
    kernel_ptxt(context, temp13, block6conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);

    
    vector<double> block6conv1summands32;
    string path13a = "/app/examples/kernel/summands/" + string("block6conv1summands32");
    txtreader(block6conv1summands32, path13a);

    // RB 5
    cout << "RB 5 ..." << endl;
    timer.start(" RB 5 ");
    vector<vector<Ciphertext>> ctxt_RB5_out;
    ctxt_RB5_out = RB(context, pack, eval, 1, ctxt_RB4_out, block6conv0multiplicands32_32_3_3, block6conv1multiplicands32_32_3_3,
    block6conv0summands32, block6conv1summands32);
    timer.end();
    cout << "DONE!" << "\n";

    //memory delete
    // ctxt_RB4_out.clear();
    // ctxt_RB4_out.shrink_to_fit();
    // block6conv0multiplicands32_32_3_3.clear();
    // block6conv0multiplicands32_32_3_3.shrink_to_fit();
    // block6conv1multiplicands32_32_3_3.claer();
    // block6conv1multiplicands32_32_3_3.shrink_to_fit();
    // block6conv0summands32.clear();
    // block6conv0summands32.shrink_to_fit();
    // block6conv1summands32.claer();
    // block6conv1summands32.shrink_to_fit();






    // Down Sampling (Residual) Block 2
        // DSB 2
    vector<double> temp14;
    vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path14 = "/app/examples/kernel/multiplicands/" + string("block7conv_onebyone_multiplicands64_32_1_1");
    txtreader(temp14, path14);
    kernel_ptxt(context, temp14, block7conv_onebyone_multiplicands64_32_1_1, 12, 2, 2, 64, 32, 1, ecd);


    vector<double> block7conv_onebyone_summands64;
    string path14a = "/app/examples/kernel/summands/" + string("block7conv_onebyone_summands64");
    txtreader(block7conv_onebyone_summands64, path14a);
    
    
    vector<double> temp15;
    vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path15 = "/app/examples/kernel/multiplicands/" + string("block7conv0multiplicands64_32_3_3");
    txtreader(temp15, path15);
    kernel_ptxt(context, temp15, block7conv0multiplicands64_32_3_3, 12, 2, 2, 64, 32, 3, ecd);

    
    vector<double> block7conv0summands64;
    string path15a = "/app/examples/kernel/summands/" + string("block7conv0summands64");
    txtreader(block7conv0summands64, path15a);
    

    vector<double> temp16;
    vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path16 = "/app/examples/kernel/multiplicands/" + string("block7conv1multiplicands64_64_3_3");
    txtreader(temp16, path16);
    kernel_ptxt(context, temp16, block7conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);
    
    vector<double> block7conv1summands64;
    string path16a = "/app/examples/kernel/summands/" + string("block7conv1summands64");
    txtreader(block7conv1summands64, path16a);

    // DSB 2
    cout << "DSB 2 ..." << endl;
    timer.start(" DSB 2 ");
    vector<vector<Ciphertext>> ctxt_DSB2_out;
    ctxt_DSB2_out = DSB(context, pack, eval, 1, ctxt_RB5_out, block7conv0multiplicands64_32_3_3, block7conv1multiplicands64_64_3_3, block7conv_onebyone_multiplicands64_32_1_1,
    block7conv0summands64, block7conv1summands64, block7conv_onebyone_summands64);
    cout << "DONE!" << "\n";

    //memory delete
    // ctxt_RB5_out.clear();
    // ctxt_RB5_out.shrink_to_fit();
    // block7conv0multiplicands64_32_3_3.clear();
    // block7conv0multiplicands64_32_3_3.shrink_to_fit();
    // block7conv1multiplicands64_64_3_3.clear();
    // block7conv1multiplicands64_64_3_3.shrink_to_fit();
    // block7conv_onebyone_multiplicands64_32_1_1.clear();
    // block7conv_onebyone_multiplicands64_32_1_1.shrink_to_fit();
    // block7conv0summands64.clear();
    // block7conv0summands64.shrink_to_fit();
    // block7conv1summands64.clear();
    // block7conv1summands64.shrink_to_fit();
    // block7conv_onebyone_summands64.clear();
    // block7conv_onebyone_summands64.shrink_to_fit();




    // Residual Block 6, 7
    // RB 6
    vector<double> temp17;
    vector<vector<vector<Plaintext>>> block8conv0multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path17 = "/app/examples/kernel/multiplicands/" + string("block8conv0multiplicands64_64_3_3");
    txtreader(temp17, path17);
    kernel_ptxt(context, temp17, block8conv0multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);

    vector<double> block8conv0summands64;
    string path17a = "/app/examples/kernel/summands/" + string("block8conv0summands64");
    txtreader(block8conv0summands64, path17a);
    
    
    vector<double> temp18;
    vector<vector<vector<Plaintext>>> block8conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path18 = "/app/examples/kernel/multiplicands/" + string("block8conv1multiplicands64_64_3_3");
    txtreader(temp18, path18);
    kernel_ptxt(context, temp18, block8conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);

    vector<double> block8conv1summands64;
    string path18a = "/app/examples/kernel/summands/" + string("block8conv1summands64");
    txtreader(block8conv0summands64, path18a);

    // RB 6
    cout << "RB 6..." << endl;
    timer.start(" RB 6 ");
    vector<vector<Ciphertext>> ctxt_RB6_out;
    ctxt_RB6_out = RB(context, pack, eval, 2, ctxt_DSB2_out, block8conv0multiplicands64_64_3_3, block8conv1multiplicands64_64_3_3,
    block8conv0summands64, block8conv1summands64);
    timer.end();
    cout << "DONE!" << "\n";


    //memory delete
    // ctxt_DSB2_out.claer();
    // ctxt_DSB2_out.shrink_to_fit();
    // block8conv0multiplicands64_64_3_3.clear();
    // block8conv0multiplicands64_64_3_3.shrink_to_fit();
    // block8conv1multiplicands64_64_3_3.clear();
    // block8conv1multiplicands64_64_3_3.shrink_to_fit();
    // block8conv0summands64.claer();
    // block8conv0summands64.shrink_to_fit();
    // block8conv1summands64.claer();
    // block8conv1summands64.shrink_to_fit();




        // RB 7
    vector<double> temp19;
    vector<vector<vector<Plaintext>>> block9conv0multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path19 = "/app/examples/kernel/multiplicands/" + string("block9conv0multiplicands64_64_3_3");
    txtreader(temp19, path19);
    kernel_ptxt(context, temp19, block9conv0multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);

    vector<double> block9conv0summands64;
    string path19a = "/app/examples/kernel/summands/" + string("block9conv0summands64");
    txtreader(block9conv0summands64, path19a);
    

    vector<double> temp20;
    vector<vector<vector<Plaintext>>> block9conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path20 = "/app/examples/kernel/multiplicands/" + string("block9conv1multiplicands64_64_3_3");
    txtreader(temp20, path20);
    kernel_ptxt(context, temp20, block9conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);
    
    vector<double> block9conv1summands64;
    string path20a = "/app/examples/kernel/summands/" + string("block9conv1summands64");
    txtreader(block9conv1summands64, path20a);

    // RB 7
    cout << "RB 7 ..." << endl;
    timer.start(" RB 7 ");
    vector<vector<Ciphertext>> ctxt_RB7_out;
    ctxt_RB7_out = RB(context, pack, eval, 2, ctxt_RB6_out, block9conv0multiplicands64_64_3_3, block9conv1multiplicands64_64_3_3,
    block9conv0summands64, block9conv1summands64);
    timer.end();
    cout << "DONE!" << "\n";

    //memory delete
    // ctxt_RB6_out.claer();
    // ctxt_RB6_out.shrink_to_fit();
    // block9conv0multiplicands64_64_3_3.claer();
    // block9conv0multiplicands64_64_3_3.shrink_to_fit();
    // block9conv1multiplicands64_64_3_3.claer();
    // block9conv1multiplicands64_64_3_3.shrink_to_fit();
    // block9conv0summands64.clear();
    // block9conv0summands64.shrink_to_fit();
    // block9conv1summands64.clear();
    // block9conv1summands64.shrink_to_fit();



    // // Average Pooling, Flatten, FC64
    // // Avg Pool
    // cout << "evaluating Avgpool" << endl;
    // timer.start("* ");
    // ctxt = Avgpool(context, pack, eval, ctxt);
    // timer.end();
    // dec.decrypt(ctxt, sk, dmsg);
    // cout << "avgpool message: " << endl;
    // printMessage(dmsg, false, 64, 64);

    // ptxt = ecd.encode(uni, 5, 0);
    // for (size_t i = 0; i < 10; ++i) {
    //     ptxt_vec.push_back(ptxt);
    // }

    // // FC64
    // cout << "evaluating FC64" << endl;
    // timer.start("* ");
    // ctxt_FC64_out = FC64old(context, pack, eval, ctxt, ptxt_vec);
    // timer.end();
    // // dec.decrypt(ctxt_out, sk, dmsg);
    // // cout << "decrypted message after FC64: " << endl;
    // // printMessage(dmsg, false, 64, 64);
    // // //(0, 8, 16, 24, 256, 264, 272, 280, 512, 520)
    // // cout << "actual result:" << endl << "[ ";
    // // cout << dmsg[0].real() << ", "<< dmsg[8].real() << ", "<< dmsg[16].real() << ", "<< dmsg[24].real() << ", "
    // // << dmsg[256].real() << ", "<< dmsg[264].real() << ", "<< dmsg[272].real() << ", "<< dmsg[280].real() << ", "
    // // << dmsg[512].real() << ", "<< dmsg[520].real() << " ]" << endl;
    // // //must be all same

    // for (size_t i = 0; i < 8; ++i) {
    //     for (size_t j = 0; j < 8; ++j) {
    //         eval.leftRotate(msg, 4*i+32*4*j, msg_tmp);
    //         eval.add(msg_tmp0, msg_tmp, msg_tmp0);
    //     }
    // }
    // eval.mult(msg_tmp0, uni, msg_tmp0);
    // for (size_t i = 0; i < 4; ++i) {
    //     for (size_t j = 0; j < 4; ++j) {
    //         for (size_t k = 0; k < 4; ++k) {
    //             eval.leftRotate(msg_tmp0, i+32*j+32*32*k, msg_tmp);
    //             eval.add(msg_out, msg_tmp, msg_out);
    //         }
    //     }
    // }
    // cout << "target value: " << endl;
    // cout << msg_out[0] << endl;

    








    
    


    // /////////////// Decryption ////////////////
    // for (int i = 0; i < 4; ++i) {
    //     for (int ch = 0; ch < 32; ++ch) {
    //         Message dmsg;
    //         cout << "Decrypt ... ";
    //         dec.decrypt(ctxt_out[i][ch], sk, dmsg);
    //         cout << "done" << endl;
    //         // printMessage(dmsg);
    //     }
    // }

    return 0;
}
