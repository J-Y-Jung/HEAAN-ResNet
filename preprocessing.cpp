#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

#include "HEaaN/heaan.hpp" 
#include "HEAAN-ResNet.hpp" 
#include "Conv.hpp"
#include "Conv_parallel.hpp"
#include "ReLUbundle.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
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

    cout.precision(7);
    
    
    
    Message msg_zero(log_slots, 0);
    
    Plaintext ptxt_zero(context);
    ptxt_zero = ecd.encode(msg_zero, 5, 0);
    
    Ciphertext ctxt_zero(context);
    enc.encrypt(msg_zero, pack, ctxt_zero, 5, 0);

    double cnst = (double)(1.0 / 40.0);
    Ciphertext ctxt_init(context);
    enc.encrypt(msg_zero, pack, ctxt_init, 0, 0);
    Plaintext ptxt_init(context);
    ptxt_init = ecd.encode(msg_zero, 0, 0);


    ///////////////////////
    /////////////////////
    /////////////////////

    
    // 1st conv
    cout << "uploading for block0conv0 ...\n\n";
    vector<double> temp0;
    vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3(16, vector<vector<Plaintext>>(3, vector<Plaintext>(9, ptxt_init)));
    string path0 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block0conv0multiplicands16_3_3_3");
    std::cout << path0 << std::endl;
    Scaletxtreader(temp0, path0, cnst);

    cout << temp0.size() <<endl;
    kernel_ptxt(context, temp0, block0conv0multiplicands16_3_3_3, 5, 1, 1, 16, 3, 3, ecd);


    temp0.clear();
    temp0.shrink_to_fit();

    vector<Plaintext> block0conv0summands16(16, ptxt_init);
    vector<double> temp0a;
    string path0a = "/app/HEAAN-ResNet/kernel/summands/" + string("block0conv0summands16");
    Scaletxtreader(temp0a, path0a, cnst);


    std::cout << temp0a.size() << std::endl;
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
            Message msg(log_slots, temp0a[i]);
            block0conv0summands16[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp0a.clear();
    temp0a.shrink_to_fit();


    #pragma omp parallel for collapse(3)
    for(int i=0; i<16; ++i){
        for(int j=0; j<3; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/")+string("/block0conv0multiplicands16_3_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block0conv0multiplicands16_3_3_3[i][j][k].save(temp);
            }
        }
    }

    //#pragma omp parallel for 
    for(int i=0; i<16; ++i){
        string temp = string("/app/parameters/summands/block0conv0summands16/")+to_string(i)+string(".bin");
        block0conv0summands16[i].save(temp);
    }
    


   

    // RB 1 - 1
    cout << "uploading for block1conv0 ...\n\n";
    vector<double> temp1;
    vector<vector<vector<Plaintext>>> block1conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path1 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block1conv0multiplicands16_16_3_3");
    txtreader(temp1, path1);
    kernel_ptxt(context, temp1, block1conv0multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp1.clear();
    temp1.shrink_to_fit();

    vector<Plaintext> block1conv0summands16(16, ptxt_init);
    vector<double> temp1a;
    string path1a = "/app/HEAAN-ResNet/kernel/summands/" + string("block1conv0summands16");
    Scaletxtreader(temp1a, path1a, cnst);
    
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        Message msg(log_slots, temp1a[i]);
        block1conv0summands16[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp1a.clear();
    temp1a.shrink_to_fit();
    
    #pragma omp parallel for collapse(3)
    for(int i=0; i<16; ++i){
        for(int j=0; j<16; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block1conv0multiplicands16_16_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block1conv0multiplicands16_16_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<16; ++i){
        string temp = string("/app/parameters/summands/block1conv0summands16/")+to_string(i)+string(".bin");
        block1conv0summands16[i].save(temp);
    }




    // RB 1 - 2
    cout<< "Uploading for block1conv1 ...\n\n";
    vector<double> temp2;
    vector<vector<vector<Plaintext>>> block1conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path2 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block1conv1multiplicands16_16_3_3/");
    txtreader(temp2, path2);
    kernel_ptxt(context, temp2, block1conv1multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp2.clear();
    temp2.shrink_to_fit();


    vector<Plaintext> block1conv1summands16(16, ptxt_init);
    vector<double> temp2a;
    string path2a = "/app/HEAAN-ResNet/kernel/summands/" + string("block1conv1summands16/");
    Scaletxtreader(temp2a, path2a, cnst);
    
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        Message msg(log_slots, temp2a[i]);
        block1conv1summands16[i] = ecd.encode(msg, 4, 0);
        }
    }

    temp2a.clear();
    temp2a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<16; ++i){
        for(int j=0; j<16; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block1conv1multiplicands16_16_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block1conv1multiplicands16_16_3_3[i][j][k].save(temp);
            }
        }
    }




    #pragma omp parallel for
    for(int i=0; i<16; ++i){
        string temp = string("/app/parameters/summands/block1conv1summands16/")+to_string(i)+string(".bin");
        block1conv1summands16[i].save(temp);
    }

  

    // RB 2 - 1
    cout << "Uploading for block2conv0...\n\n";
    vector<double> temp3;
    vector<vector<vector<Plaintext>>> block2conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path3 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block2conv0multiplicands16_16_3_3");
    txtreader(temp3, path3);
    kernel_ptxt(context, temp3, block2conv0multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp3.clear();
    temp3.shrink_to_fit();


    vector<Plaintext> block2conv0summands16(16, ptxt_init);
    vector<double> temp3a;
    string path3a = "/app/HEAAN-ResNet/kernel/summands/" + string("block2conv0summands16");
    Scaletxtreader(temp3a, path3a, cnst);
    std::cout << "here" << std::endl;
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        Message msg(log_slots, temp3a[i]);
        block2conv0summands16[i] = ecd.encode(msg, 4, 0);
        }
    }
    temp3a.clear();
    temp3a.shrink_to_fit();
    std::cout << "here2" << std::endl;


    #pragma omp parallel for collapse(3)
    for(int i=0; i<16; ++i){
        for(int j=0; j<16; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block2conv0multiplicands16_16_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block2conv0multiplicands16_16_3_3[i][j][k].save(temp);
            }
        }
    }



    #pragma omp parallel for
    for(int i=0; i<16; ++i){
        string temp = string("/app/parameters/summands/block2conv0summands16/")+to_string(i)+string(".bin");
        block2conv0summands16[i].save(temp);
    }



    // RB 2 - 2
    vector<double> temp4;
    vector<vector<vector<Plaintext>>> block2conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path4 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block2conv1multiplicands16_16_3_3");
    txtreader(temp4, path4);
    kernel_ptxt(context, temp4, block2conv1multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp4.clear();
    temp4.shrink_to_fit();


    vector<Plaintext> block2conv1summands16(16, ptxt_init);
    vector<double> temp4a;
    string path4a = "/app/HEAAN-ResNet/kernel/summands/" + string("block2conv1summands16");
    Scaletxtreader(temp4a, path4a, cnst);

    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        Message msg(log_slots, temp4a[i]);
        block2conv1summands16[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp4a.clear();
    temp4a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<16; ++i){
        for(int j=0; j<16; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block2conv1multiplicands16_16_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block2conv1multiplicands16_16_3_3[i][j][k].save(temp);
            }
        }
    }




    #pragma omp parallel for
    for(int i=0; i<16; ++i){
        string temp = string("/app/parameters/summands/block2conv1summands16/")+to_string(i)+string(".bin");
        block2conv1summands16[i].save(temp);
    }


    // RB 3 - 1
    vector<double> temp5;
    vector<vector<vector<Plaintext>>> block3conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path5 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block3conv0multiplicands16_16_3_3");
    txtreader(temp5, path5);
    kernel_ptxt(context, temp5, block3conv0multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp5.clear();
    temp5.shrink_to_fit();


    vector<Plaintext> block3conv0summands16(16, ptxt_init);
    vector<double> temp5a;
    string path5a = "/app/HEAAN-ResNet/kernel/summands/" + string("block3conv0summands16");
    Scaletxtreader(temp5a, path5a, cnst);

    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        Message msg(log_slots, temp5a[i]);
        block3conv0summands16[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp5a.clear();
    temp5a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<16; ++i){
        for(int j=0; j<16; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block3conv0multiplicands16_16_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block3conv0multiplicands16_16_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for
    for(int i=0; i<16; ++i){
        string temp = string("/app/parameters/summands/block3conv0summands16/")+to_string(i)+string(".bin");
        block3conv0summands16[i].save(temp);
    }




    // RB 3 - 2
    vector<double> temp6;
    vector<vector<vector<Plaintext>>> block3conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path6 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block3conv1multiplicands16_16_3_3");
    txtreader(temp6, path6);
    kernel_ptxt(context, temp6, block3conv1multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp6.clear();
    temp6.shrink_to_fit();


    vector<Plaintext> block3conv1summands16(16, ptxt_init);
    vector<double> temp6a;
    string path6a = "/app/HEAAN-ResNet/kernel/summands/" + string("block3conv1summands16");
    Scaletxtreader(temp6a, path6a, cnst);
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        Message msg(log_slots, temp6a[i]);
        block3conv1summands16[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp6a.clear();
    temp6a.shrink_to_fit();
    #pragma omp parallel for collapse(3)
    for(int i=0; i<16; ++i){
        for(int j=0; j<3; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block3conv1multiplicands16_16_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block3conv1multiplicands16_16_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<16; ++i){
        string temp = string("/app/parameters/summands/block3conv1summands16/")+to_string(i)+string(".bin");
        block3conv1summands16[i].save(temp);
    }



    // DSB 1 - res
    vector<double> temp7;
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1(32, vector<vector<Plaintext>>(16, vector<Plaintext>(1, ptxt_init)));
    string path7 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block4conv_onebyone_multiplicands32_16_1_1");
    txtreader(temp7, path7);
    kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 5, 1, 2, 32, 16, 1, ecd);
    temp7.clear();
    temp7.shrink_to_fit();


    vector<Plaintext> block4conv_onebyone_summands32(32, ptxt_init);
    vector<double> temp7a;
    string path7a = "/app/HEAAN-ResNet/kernel/summands/" + string("block4conv_onebyone_summands32");
    Scaletxtreader(temp7a, path7a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 32; ++i) {
        #pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp7a[i]);
        block4conv_onebyone_summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp7a.clear();
    temp7a.shrink_to_fit();
    #pragma omp parallel for collapse(3)
    for(int i=0; i<32; ++i){
        for(int j=0; j<16; ++j){
            for(int k=0; k<1; ++k){
                string temp = string("/app/parameters/multiplicands/block4conv_onebyone_multiplicands32_16_1_1/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block4conv_onebyone_multiplicands32_16_1_1[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<32; ++i){
        string temp = string("/app/parameters/summands/block4conv_onebyone_summands32/")+to_string(i)+string(".bin");
        block4conv_onebyone_summands32[i].save(temp);
    }



    // DSB 1 - 1
    vector<double> temp8;
    vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path8 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block4conv0multiplicands32_16_3_3");
    txtreader(temp8, path8);
    kernel_ptxt(context, temp8, block4conv0multiplicands32_16_3_3, 5, 1, 2, 32, 16, 3, ecd);
    temp8.clear();
    temp8.shrink_to_fit();


    vector<Plaintext> block4conv0summands32(32, ptxt_init);
    vector<double> temp8a;
    string path8a = "/app/HEAAN-ResNet/kernel/summands/" + string("block4conv0summands32");
    Scaletxtreader(temp8a, path8a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 32; ++i) {
        #pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp8a[i]);
        block4conv0summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp8a.clear();
    temp8a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<32; ++i){
        for(int j=0; j<16; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block4conv0multiplicands32_16_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block4conv0multiplicands32_16_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<32; ++i){
        string temp = string("/app/parameters/summands/block4conv0summands32/")+to_string(i)+string(".bin");
        block4conv0summands32[i].save(temp);
    }


    
    // DSB 1 - 2
    vector<double> temp9;
    vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path9 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block4conv1multiplicands32_32_3_3");
    txtreader(temp9, path9);
    kernel_ptxt(context, temp9, block4conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp9.clear();
    temp9.shrink_to_fit();


    vector<Plaintext> block4conv1summands32(32, ptxt_init);
    vector<double> temp9a;
    string path9a = "/app/HEAAN-ResNet/kernel/summands/" + string("block4conv1summands32");
    Scaletxtreader(temp9a, path9a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 32; ++i) {
        #pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp9a[i]);
        block4conv1summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp9a.clear();
    temp9a.shrink_to_fit();
    #pragma omp parallel for collapse(3)
    for(int i=0; i<32; ++i){
        for(int j=0; j<32; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block4conv1multiplicands32_32_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block4conv1multiplicands32_32_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<32; ++i){
        string temp = string("/app/parameters/summands/block4conv1summands32/")+to_string(i)+string(".bin");
        block4conv1summands32[i].save(temp);
    }


        // RB 4 - 1
    vector<double> temp10;
    vector<vector<vector<Plaintext>>> block5conv0multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path10 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block5conv0multiplicands32_32_3_3");
    txtreader(temp10, path10);
    kernel_ptxt(context, temp10, block5conv0multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp10.clear();
    temp10.shrink_to_fit();


    vector<Plaintext> block5conv0summands32(32, ptxt_init);
    vector<double> temp10a;
    string path10a = "/app/HEAAN-ResNet/kernel/summands/" + string("block5conv0summands32");
    Scaletxtreader(temp10a, path10a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 32; ++i) {
        #pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp10a[i]);
        block5conv0summands32[i]=ecd.encode(msg, 4, 0);
        }
    }

    
    temp10a.clear();
    temp10a.shrink_to_fit();
    

    #pragma omp parallel for collapse(3)
    for(int i=0; i<32; ++i){
        for(int j=0; j<32; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block5conv0multiplicands32_32_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block5conv0multiplicands32_32_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<32; ++i){
        string temp = string("/app/parameters/summands/block5conv0summands32/")+to_string(i)+string(".bin");
        block5conv0summands32[i].save(temp);
    }



    // RB 4 - 2
    vector<double> temp11;
    vector<vector<vector<Plaintext>>> block5conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path11 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block5conv1multiplicands32_32_3_3");
    txtreader(temp11, path11);
    kernel_ptxt(context, temp11, block5conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp11.clear();
    temp11.shrink_to_fit();


    vector<Plaintext> block5conv1summands32(32, ptxt_init);
    vector<double> temp11a;
    string path11a = "/app/HEAAN-ResNet/kernel/summands/" + string("block5conv1summands32");
    Scaletxtreader(temp11a, path11a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 32; ++i) {
        #pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp11a[i]);
        block5conv1summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp11a.clear();
    temp11a.shrink_to_fit();


    #pragma omp parallel for collapse(3)
    for(int i=0; i<32; ++i){
        for(int j=0; j<32; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block5conv1multiplicands32_32_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block5conv1multiplicands32_32_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<32; ++i){
        string temp = string("/app/parameters/summands/block5conv1summands32/")+to_string(i)+string(".bin");
        block5conv1summands32[i].save(temp);
    }


    // RB 5 - 1
    vector<double> temp12;
    vector<vector<vector<Plaintext>>> block6conv0multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path12 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block6conv0multiplicands32_32_3_3");
    txtreader(temp12, path12);
    kernel_ptxt(context, temp12, block6conv0multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp12.clear();
    temp12.shrink_to_fit();

    vector<Plaintext> block6conv0summands32(32, ptxt_init);
    vector<double> temp12a;
    string path12a = "/app/HEAAN-ResNet/kernel/summands/" + string("block6conv0summands32");
    Scaletxtreader(temp12a, path12a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 32; ++i) {
        #pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp12a[i]);
        block6conv0summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp12a.clear();
    temp12a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<32; ++i){
        for(int j=0; j<32; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block6conv0multiplicands32_32_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block6conv0multiplicands32_32_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<32; ++i){
        string temp = string("/app/parameters/summands/block6conv0summands32/")+to_string(i)+string(".bin");
        block6conv0summands32[i].save(temp);
    }



    // RB 5 - 2
    vector<double> temp13;
    vector<vector<vector<Plaintext>>> block6conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path13 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block6conv1multiplicands32_32_3_3");
    txtreader(temp13, path13);
    kernel_ptxt(context, temp13, block6conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp13.clear();
    temp13.shrink_to_fit();


    vector<Plaintext> block6conv1summands32(32, ptxt_init);
    vector<double> temp13a;
    string path13a = "/app/HEAAN-ResNet/kernel/summands/" + string("block6conv1summands32");
    Scaletxtreader(temp13a, path13a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 32; ++i) {
        #pragma omp parallel num_threads(2)
        {
        Message msg(log_slots, temp13a[i]);
        block6conv1summands32[i]=ecd.encode(msg, 4, 0);
        }
    }
    temp13a.clear();
    temp13a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<32; ++i){
        for(int j=0; j<32; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block6conv1multiplicands32_32_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block6conv1multiplicands32_32_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<32; ++i){
        string temp = string("/app/parameters/summands/block6conv1summands32/")+to_string(i)+string(".bin");
        block6conv1summands32[i].save(temp);
    }




    // Down Sampling (Residual) Block 2
    // DSB 2



    ////////////////////////
    /////// DSB 2 //////////
    /////////////////////////


    ///////////////////// Residual flow ////////////////////////////
    // Convolution


    vector<double> temp14;
    vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1(64, vector<vector<Plaintext>>(32, vector<Plaintext>(1, ptxt_init)));
    string path14 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block7conv_onebyone_multiplicands64_32_1_1");
    txtreader(temp14, path14);
    kernel_ptxt(context, temp14, block7conv_onebyone_multiplicands64_32_1_1, 5, 2, 2, 64, 32, 1, ecd);
    temp14.clear();
    temp14.shrink_to_fit();

    vector<Plaintext> block7conv_onebyone_summands64(64, ptxt_init);
    vector<double> temp14a;
    string path14a = "/app/HEAAN-ResNet/kernel/summands/" + string("block7conv_onebyone_summands64");
    Scaletxtreader(temp14a, path14a, cnst);

    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp14a[i]);
        block7conv_onebyone_summands64[i]=ecd.encode(msg, 4, 0);
    }
    temp14a.clear();
    temp14a.shrink_to_fit();


    #pragma omp parallel for collapse(3)
    for(int i=0; i<64; ++i){
        for(int j=0; j<32; ++j){
            for(int k=0; k<1; ++k){
                string temp = string("/app/parameters/multiplicands/block7conv_onebyone_multiplicands64_32_1_1/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block7conv_onebyone_multiplicands64_32_1_1[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<64; ++i){
        string temp = string("/app/parameters/summands/block7conv_onebyone_summands64/")+to_string(i)+string(".bin");
        block7conv_onebyone_summands64[i].save(temp);
    }


//////////

    vector<double> temp15;
    vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path15 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block7conv0multiplicands64_32_3_3");
    txtreader(temp15, path15);
    kernel_ptxt(context, temp15, block7conv0multiplicands64_32_3_3, 5, 2, 2, 64, 32, 3, ecd);
    temp15.clear();
    temp15.shrink_to_fit();


    vector<Plaintext> block7conv0summands64(64, ptxt_init);
    vector<double> temp15a;
    string path15a = "/app/HEAAN-ResNet/kernel/summands/" + string("block7conv0summands64");
    Scaletxtreader(temp15a, path15a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp15a[i]);
        block7conv0summands64[i]=ecd.encode(msg, 4, 0);
    }
    temp15a.clear();
    temp15a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<64; ++i){
        for(int j=0; j<32; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block7conv0multiplicands64_32_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block7conv0multiplicands64_32_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<64; ++i){
        string temp = string("/app/parameters/summands/block7conv0summands64/")+to_string(i)+string(".bin");
        block7conv0summands64[i].save(temp);
    }


    // Second convolution

    vector<double> temp16;
    vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path16 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block7conv1multiplicands64_64_3_3");
    txtreader(temp16, path16);
    kernel_ptxt(context, temp16, block7conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    temp16.clear();
    temp16.shrink_to_fit();

    vector<Plaintext> block7conv1summands64(64, ptxt_init);
    vector<double> temp16a;
    string path16a = "/app/HEAAN-ResNet/kernel/summands/" + string("block7conv1summands64");
    Scaletxtreader(temp16a, path16a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp16a[i]);
        block7conv1summands64[i]=ecd.encode(msg, 4, 0);
    }

    temp16a.clear();
    temp16a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<64; ++i){
        for(int j=0; j<64; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block7conv1multiplicands64_64_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block7conv1multiplicands64_64_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<64; ++i){
        string temp = string("/app/parameters/summands/block7conv1summands64/")+to_string(i)+string(".bin");
        block7conv1summands64[i].save(temp);
    }



    /////////////////////////////////////
    ////////////// RB 6 //////////////////
    /////////////////////////////////////


    ///////////////////////// Main flow /////////////////////////////////////////
    vector<double> temp17;
    vector<vector<vector<Plaintext>>> block8conv0multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path17 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block8conv0multiplicands64_64_3_3");
    txtreader(temp17, path17);
    kernel_ptxt(context, temp17, block8conv0multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    temp17.clear();
    temp17.shrink_to_fit();

    vector<Plaintext> block8conv0summands64(64, ptxt_init);
    vector<double> temp17a;
    string path17a = "/app/HEAAN-ResNet/kernel/summands/" + string("block8conv0summands64");
    Scaletxtreader(temp17a, path17a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp17a[i]);
        block8conv0summands64[i]=ecd.encode(msg, 4, 0);
    }
    temp17a.clear();
    temp17a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<64; ++i){
        for(int j=0; j<64; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block8conv0multiplicands64_64_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block8conv0multiplicands64_64_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<64; ++i){
        string temp = string("/app/parameters/summands/block8conv0summands64/")+to_string(i)+string(".bin");
        block8conv0summands64[i].save(temp);
    }



    // Second convolution



    vector<double> temp18;
    vector<vector<vector<Plaintext>>> block8conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path18 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block8conv1multiplicands64_64_3_3");
    txtreader(temp18, path18);
    kernel_ptxt(context, temp18, block8conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    temp18.clear();
    temp18.shrink_to_fit();

    vector<Plaintext> block8conv1summands64(64, ptxt_init);
    vector<double> temp18a;
    string path18a = "/app/HEAAN-ResNet/kernel/summands/" + string("block8conv1summands64");
    Scaletxtreader(temp18a, path18a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp18a[i]);
        block8conv1summands64[i]=ecd.encode(msg, 4, 0);
    }

    temp18a.clear();
    temp18a.shrink_to_fit();


    #pragma omp parallel for collapse(3)
    for(int i=0; i<64; ++i){
        for(int j=0; j<64; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block8conv1multiplicands64_64_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block8conv1multiplicands64_64_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<64; ++i){
        string temp = string("/app/parameters/summands/block8conv1summands64/")+to_string(i)+string(".bin");
        block8conv1summands64[i].save(temp);
    }





    /////////////////////////
    //////////// RB 7//////////////
    ////////////////////////////




    ///////////////////////// Main flow /////////////////////////////////////////

    vector<double> temp19;
    vector<vector<vector<Plaintext>>> block9conv0multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path19 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block9conv0multiplicands64_64_3_3");
    txtreader(temp19, path19);
    kernel_ptxt(context, temp19, block9conv0multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    temp19.clear();
    temp19.shrink_to_fit();

    vector<Plaintext> block9conv0summands64(64, ptxt_init);
    vector<double> temp19a;
    string path19a = "/app/HEAAN-ResNet/kernel/summands/" + string("block9conv0summands64");
    Scaletxtreader(temp19a, path19a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp19a[i]);
        block9conv0summands64[i]=ecd.encode(msg, 4, 0);
    }

    temp19a.clear();
    temp19a.shrink_to_fit();


    #pragma omp parallel for collapse(3)
    for(int i=0; i<64; ++i){
        for(int j=0; j<64; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block9conv0multiplicands64_64_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block9conv0multiplicands64_64_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<64; ++i){
        string temp = string("/app/parameters/summands/block9conv0summands64/")+to_string(i)+string(".bin");
        block9conv0summands64[i].save(temp);
    }



    // Second convolution

    vector<double> temp20;
    vector<vector<vector<Plaintext>>> block9conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path20 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block9conv1multiplicands64_64_3_3");
    txtreader(temp20, path20);
    kernel_ptxt(context, temp20, block9conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);

    temp20.clear();
    temp20.shrink_to_fit();

    vector<Plaintext> block9conv1summands64(64, ptxt_init);
    vector<double> temp20a;
    string path20a = "/app/HEAAN-ResNet/kernel/summands/" + string("block9conv1summands64");
    Scaletxtreader(temp20a, path20a, cnst);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp20a[i]);
        block9conv1summands64[i]=ecd.encode(msg, 4, 0);
    }

    temp20a.clear();
    temp20a.shrink_to_fit();

    #pragma omp parallel for collapse(3)
    for(int i=0; i<64; ++i){
        for(int j=0; j<64; ++j){
            for(int k=0; k<9; ++k){
                string temp = string("/app/parameters/multiplicands/block9conv1multiplicands64_64_3_3/") +to_string(i)+string("_")+to_string(j)+string("_")+to_string(k)+string(".bin");
                block9conv1multiplicands64_64_3_3[i][j][k].save(temp);
            }
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<64; ++i){
        string temp = string("/app/parameters/summands/block9conv1summands64/")+to_string(i)+string(".bin");
        block9conv1summands64[i].save(temp);
    }

    //FC64 setup...
    vector<double> temp21;
    vector<vector<Plaintext>> fclayermultiplicands10_64(10, vector<Plaintext>(64, ptxt_init));
    string path21 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("fclayermultiplicands10_64");
    double cnst2 = (double)(1.0 / 64.0);
    Scaletxtreader(temp21, path21,cnst2);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            Message msg(log_slots, temp21[64 * i + j]);
            fclayermultiplicands10_64[i][j] = ecd.encode(msg, 1, 0);
        }
    }

    temp21.clear();
    temp21.shrink_to_fit();

    vector<double> temp21a;
    vector<Plaintext> fclayersummands10(10, ptxt_init);
    string path21a = "/app/HEAAN-ResNet/kernel/summands/" + string("fclayersummands10");

    double cnst1 = (double)(1.0 / 40.0);
    Scaletxtreader(temp21a, path21a, cnst1);
    
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 10; ++i) {
        #pragma omp parallel num_threads(8)
        {
        Message msg(log_slots, temp21a[i]);
        fclayersummands10[i] = ecd.encode(msg, 0, 0);
        }
    }

    temp21a.clear();
    temp21a.shrink_to_fit();
    

    #pragma omp parallel for collapse(3)
    for(int i=0; i<10; ++i){
        for(int j=0; j<64; ++j){
            string temp = string("/app/parameters/multiplicands/fclayermultiplicands10_64/") +to_string(i)+string("_")+to_string(j)+string(".bin");
            fclayermultiplicands10_64[i][j].save(temp);
        }
    }

    #pragma omp parallel for 
    for(int i=0; i<10; ++i){
        string temp = string("/app/parameters/summands/fclayersummands10/")+to_string(i)+string(".bin");
        fclayersummands10[i].save(temp);
    }



    
    return 0;
}
