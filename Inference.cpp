#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "HEaaN/heaan.hpp" // 필요한가?
#include "examples.hpp" // 필요한가?
#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "DSB.hpp"
#include "convtools.hpp"
#include "kernelEncode.hpp"

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
        cout << "Bootstrap is not available for parameter "
            << presetNamer(preset) << endl;
        return -1;
    }
    cout << "Parameter : " << presetNamer(preset) << endl
        << endl;
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




    ///////////// Message ///////////////////

    // 1st conv
    vector<double> temp0;
    vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3;
    string path0 = "./kernel/multiplicands/" + "block0conv0multiplicands16_3_3_3";
    txtreader(temp0, path0);
    kernel_ptxt(context, temp0, block0conv0multiplicands16_3_3_3, 12, 1, 1, 16, 3, 3, ecd);
    
    vector<double> block0conv0summands16;
    string path0a = "./kernel/summands/" + "block0conv0summands16";
    txtreader(block0conv0summands16, path0a);


    // Convolution 1
    cout << "Convolution 1 ..." << endl;
    timer.start(" Convolution 1 ");
    vector<vector<Ciphertext>> ctxt_conv1_out_bundle;
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        vector<Ciphertext> ctxt_conv1_out_cache;
        ctxt_conv1_out_cache = Conv(context, pack, eval, 32, 1, 1, 3, 16, ctxt_bundle[i], block0conv0multiplicands16_3_3_3);
        ctxt_conv1_out_bundle.push_back(ctxt_conv1_out_cache);
    }
    addBNsummands(context, ctxt_conv1_out_bundle, block0conv0summands16, 16, 16);

    timer.end();
    cout << "DONE!" << "\n";


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
    vector<vector<vector<Plaintext>>> block1conv0multiplicands16_16_3_3;
    string path1 = "./kernel/multiplicands/" + "block1conv0multiplicands16_16_3_3";
    txtreader(temp1, path1);
    kernel_ptxt(context, temp1, block1conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);
    
    vector<double> block1conv0summands16;
    string path1a = "./kernel/summands/" + "block1conv0summands16";
    txtreader(block1conv0summands16, path1a);
    
        // RB 1 - 2
    vector<double> temp2;
    vector<vector<vector<Plaintext>>> block1conv1multiplicands16_16_3_3;
    string path2 = "./kernel/multiplicands/" + "block1conv1multiplicands16_16_3_3";
    txtreader(temp2, path2);
    kernel_ptxt(context, temp2, block1conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block1conv1summands16;
    string path2a = "./kernel/summands/" + "block1conv1summands16";
    txtreader(block0conv0summands16, path2a);

    // RB 1
    cout << "RB 1 ..." << endl;
    timer.start(" RB 1 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB1_out = RB(context, pack, eval, 0, ctxt_relu1_out_bundle, block1conv0multiplicands16_16_3_3, block1conv1multiplicands16_16_3_3
    block1conv0summands16, block1conv1summands16);
    timer.end();
    cout << "DONE!" << "\n";




        // RB 2 - 1
    vector<double> temp3;
    vector<vector<vector<Plaintext>>> block2conv0multiplicands16_16_3_3;
    string path3 = "./kernel/multiplicands/" + "block2conv0multiplicands16_16_3_3";
    txtreader(temp3, path3);
    kernel_ptxt(context, temp3, block2conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block2conv0summands16;
    string path3a = "./kernel/summands/" + "block2conv0summands16";
    txtreader(block2conv0summands16, path3a);
    
    // RB 2 - 2
    vector<double> temp4;
    vector<vector<vector<Plaintext>>> block2conv1multiplicands16_16_3_3;
    string path4 = "./kernel/multiplicands/" + "block2conv1multiplicands16_16_3_3";
    txtreader(temp4, path4);
    kernel_ptxt(context, temp4, block2conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block2conv1summands16;
    string path4a = "./kernel/summands/" + "block2conv1summands16";
    txtreader(block2conv1summands16, path4a);




    // RB 2
    cout << "RB 2..." << endl;
    timer.start(" RB 2 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB2_out = RB(context, pack, eval, 0, ctxt_RB1_out, block2conv0multiplicands16_16_3_3, block2conv1multiplicands16_16_3_3);
    timer.end();
    cout << "DONE!" << "\n";

















        // RB 3 - 1
    vector<double> temp5;
    vector<vector<vector<Plaintext>>> block3conv0multiplicands16_16_3_3;
    string path5 = "./kernel/multiplicands/" + "block3conv0multiplicands16_16_3_3";
    txtreader(temp5, path5);
    kernel_ptxt(context, temp5, block3conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block3conv0summands16;
    string path5a = "./kernel/summands/" + "block3conv0summands16";
    txtreader(block3conv0summands16, path5a);
    
    
    // RB 3 - 2
    vector<double> temp6;
    vector<vector<vector<Plaintext>>> block3conv1multiplicands16_16_3_3;
    string path6 = "./kernel/multiplicands/" + "block3conv1multiplicands16_16_3_3";
    txtreader(temp6, path6);
    kernel_ptxt(context, temp6, block3conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> block3conv1summands16;
    string path6a = "./kernel/summands/" + "block3conv1summands16";
    txtreader(block3conv1summands16, path5a);

    // RB 3
    cout << "RB 3 ..." << endl;
    timer.start(" RB 3 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB3_out = RB(context, pack, eval, 0, ctxt_RB2_out, block3conv0multiplicands16_16_3_3, block3conv1multiplicands16_16_3_3);
    timer.end();
    cout << "DONE!" << "\n";


    // Down Sampling (Residual) Block 1
        // DSB 1 - res
    vector<double> temp7;
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1;
    string path7 = "./kernel/multiplicands/" + "block4conv_onebyone_multiplicands32_16_1_1";
    txtreader(temp7, path7);
    kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 12, 1, 2, 32, 16, 1, ecd);

    
    vector<double> block4conv_onebyone_summands32;
    string path7a = "./kernel/summands/" + "block4conv_onebyone_summands16";
    txtreader(block4conv_onebyone_conv1summands32, path7a);
    
    // DSB 1 - 1
    vector<double> temp8;
    vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3;
    string path8 = "./kernel/multiplicands/" + "block4conv0multiplicands32_16_3_3";
    txtreader(temp8, path8);
    kernel_ptxt(context, temp8, block4conv0multiplicands32_16_3_3, 12, 1, 2, 32, 16, 3, ecd);


    vector<double> block4conv0summands32;
    string path8a = "./kernel/summands/" + "block4conv0summands32";
    txtreader(block4conv0summands32, path8a);
    
    
    // DSB 1 - 2
    vector<double> temp9;
    vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3;
    string path9 = "./kernel/multiplicands/" + "block4conv1multiplicands32_32_3_3";
    txtreader(temp9, path9);
    kernel_ptxt(context, temp9, block4conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);


    vector<double> block4conv1summands32;
    string path9a = "./kernel/summands/" + "block4conv1summands32";
    txtreader(block4conv1summands32, path9a);

    // DSB 1
    cout << "DSB 1 ..." << endl;
    timer.start(" DSB 1 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_DSB1_out = DSB(timer, context, pack, eval, 0, ctxt_RB3_out, block4conv0multiplicands32_16_3_3, block4conv1multiplicands32_32_3_3, block4conv_onebyone_multiplicands32_16_1_1);
    cout << "DONE!" << "\n";


    // Residual Block 4, 5
        // RB 4 - 1
    vector<double> temp10;
    vector<vector<vector<Plaintext>>> block5conv0multiplicands32_32_3_3;
    string path10 = "./kernel/multiplicands/" + "block5conv0multiplicands32_32_3_3";
    txtreader(temp10, path10);
    kernel_ptxt(context, temp10, block5conv0multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);
    
    
    vector<double> block5conv0summands32;
    string path10a = "./kernel/summands/" + "block5conv0summands32";
    txtreader(block5conv0summands32, path10a);
    
    
    // RB 4 - 2
    vector<double> temp11;
    vector<vector<vector<Plaintext>>> block5conv1multiplicands32_32_3_3;
    string path1 = "./kernel/multiplicands/" + "block5conv1multiplicands32_32_3_3";
    txtreader(temp11, path11);
    kernel_ptxt(context, temp11, block5conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);
    
    
    vector<double> block5conv1summands32;
    string path11a = "./kernel/summands/" + "block5conv1summands32";
    txtreader(block5conv1summands32, path11a);

    // RB 4
    cout << "RB 4..." << endl;
    timer.start(" RB 4 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB4_out = RB(context, pack, eval, 1, ctxt_DSB1_out, block5conv0multiplicands32_32_3_3, block5conv1multiplicands32_32_3_3);
    timer.end();
    cout << "DONE!" << "\n";















        // RB 5 - 1
    vector<double> temp12;
    vector<vector<vector<Plaintext>>> block6conv0multiplicands32_32_3_3;
    string path12 = "./kernel/multiplicands/" + "block6conv0multiplicands32_32_3_3";
    txtreader(temp12, path12);
    kernel_ptxt(context, temp12, block6conv0multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);

    vector<double> block6conv0summands32;
    string path12a = "./kernel/summands/" + "block6conv0summands32";
    txtreader(block6conv0summands32, path12a);
    
    // RB 5 - 2
    vector<double> temp13;
    vector<vector<vector<Plaintext>>> block6conv1multiplicands32_32_3_3;
    string path13 = "./kernel/multiplicands/" + "block6conv1multiplicands32_32_3_3";
    txtreader(temp13, path13);
    kernel_ptxt(context, temp13, block6conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);

    
    vector<double> block6conv1summands32;
    string path13a = "./kernel/summands/" + "block6conv1summands32";
    txtreader(block6conv1summands32, path13a);

    // RB 5
    cout << "RB 5 ..." << endl;
    timer.start(" RB 5 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB5_out = RB(context, pack, eval, 1, ctxt_RB4_out, block6conv0multiplicands32_32_3_3, block6conv1multiplicands32_32_3_3);
    timer.end();
    cout << "DONE!" << "\n";






    // Down Sampling (Residual) Block 2
        // DSB 2
    vector<double> temp14;
    vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1;
    string path14 = "./kernel/multiplicands/" + "block7conv_onebyone_multiplicands64_32_1_1";
    txtreader(temp14, path14);
    kernel_ptxt(context, temp14, block7conv_onebyone_multiplicands64_32_1_1, 12, 2, 2, 64, 32, 1, ecd);


    vector<double> block7conv_onebyone_summands64;
    string path14a = "./kernel/summands/" + "block7conv_onebyone_summands64";
    txtreader(block7conv_onebyone_summands64, path14a);
    
    
    vector<double> temp15;
    vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3;
    string path15 = "./kernel/multiplicands/" + "block7conv0multiplicands64_32_3_3";
    txtreader(temp15, path15);
    kernel_ptxt(context, temp15, block7conv0multiplicands64_32_3_3, 12, 2, 2, 64, 32, 3, ecd);

    
    vector<double> block7conv0summands64;
    string path15a = "./kernel/summands/" + "block7conv0summands64";
    txtreader(block7conv0summands64, path15a);
    

    vector<double> temp16;
    vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3;
    string path16 = "./kernel/multiplicands/" + "block7conv1multiplicands64_64_3_3";
    txtreader(temp16, path16);
    kernel_ptxt(context, temp16, block7conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);
    
    vector<double> block7conv1summands64;
    string path16a = "./kernel/summands/" + "block7conv1summands64";
    txtreader(block7conv1summands64, path16a);

    // DSB 2
    cout << "DSB 2 ..." << endl;
    timer.start(" DSB 2 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_DSB2_out = DSB(timer, context, pack, eval, 1, ctxt_RB5_out, block7conv0multiplicands64_32_3_3, block7conv1multiplicands64_64_3_3, block7conv_onebyone_multiplicands64_32_1_1);
    cout << "DONE!" << "\n";


    // Residual Block 6, 7
    // RB 6
    vector<double> temp17;
    vector<vector<vector<Plaintext>>> block8conv0multiplicands64_64_3_3;
    string path17 = "./kernel/multiplicands/" + "block8conv0multiplicands64_64_3_3";
    txtreader(temp17, path17);
    kernel_ptxt(context, temp17, block8conv0multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);

    vector<double> block8conv0summands64;
    string path17a = "./kernel/summands/" + "block8conv0summands64";
    txtreader(block8conv0summands64, path17a);
    
    
    vector<double> temp18;
    vector<vector<vector<Plaintext>>> block8conv1multiplicands64_64_3_3;
    string path18 = "./kernel/multiplicands/" + "block8conv1multiplicands64_64_3_3";
    txtreader(temp18, path18);
    kernel_ptxt(context, temp18, block8conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);

    vector<double> block8conv1summands64;
    string path18a = "./kernel/summands/" + "block8conv1summands64";
    txtreader(block8conv0summands64, path18a);

    // RB 6
    cout << "RB 6..." << endl;
    timer.start(" RB 6 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB6_out = RB(context, pack, eval, 2, ctxt_DSB2_out, block8conv0multiplicands64_64_3_3, block8conv1multiplicands64_64_3_3);
    timer.end();
    cout << "DONE!" << "\n";




        // RB 7
    vector<double> temp19;
    vector<vector<vector<Plaintext>>> block9conv0multiplicands64_64_3_3;
    string path19 = "./kernel/multiplicands/" + "block9conv0multiplicands64_64_3_3";
    txtreader(temp19, path19);
    kernel_ptxt(context, temp19, block9conv0multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);

    vector<double> block9conv0summands64;
    string path19a = "./kernel/summands/" + "block9conv0summands64";
    txtreader(block9conv0summands64, path19a);
    

    vector<double> temp20;
    vector<vector<vector<Plaintext>>> block9conv1multiplicands64_64_3_3;
    string path20 = "./kernel/multiplicands/" + "block9conv1multiplicands64_64_3_3";
    txtreader(temp20, path20);
    kernel_ptxt(context, temp21, block9conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);
    
    vector<double> block9conv1summands64;
    string path20a = "./kernel/summands/" + "block9conv1summands64";
    txtreader(block9conv1summands64, path20a);

    // RB 7
    cout << "RB 7 ..." << endl;
    timer.start(" RB 7 ");
    vector<vector<Ciphertext>> ctxt_out;
    ctxt_RB7_out = RB(context, pack, eval, 2, ctxt_RB6_out, block9conv0multiplicands64_64_3_3, block9conv1multiplicands64_64_3_3);
    timer.end();
    cout << "DONE!" << "\n";


    // Average Pooling, Flatten, FC64
    // Avg Pool
    cout << "evaluating Avgpool" << endl;
    timer.start("* ");
    ctxt = Avgpool(context, pack, eval, ctxt);
    timer.end();
    dec.decrypt(ctxt, sk, dmsg);
    cout << "avgpool message: " << endl;
    printMessage(dmsg, false, 64, 64);

    ptxt = ecd.encode(uni, 5, 0);
    for (size_t i = 0; i < 10; ++i) {
        ptxt_vec.push_back(ptxt);
    }

    // FC64
    cout << "evaluating FC64" << endl;
    timer.start("* ");
    ctxt_FC64_out = FC64old(context, pack, eval, ctxt, ptxt_vec);
    timer.end();
    // dec.decrypt(ctxt_out, sk, dmsg);
    // cout << "decrypted message after FC64: " << endl;
    // printMessage(dmsg, false, 64, 64);
    // //(0, 8, 16, 24, 256, 264, 272, 280, 512, 520)
    // cout << "actual result:" << endl << "[ ";
    // cout << dmsg[0].real() << ", "<< dmsg[8].real() << ", "<< dmsg[16].real() << ", "<< dmsg[24].real() << ", "
    // << dmsg[256].real() << ", "<< dmsg[264].real() << ", "<< dmsg[272].real() << ", "<< dmsg[280].real() << ", "
    // << dmsg[512].real() << ", "<< dmsg[520].real() << " ]" << endl;
    // //must be all same

    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            eval.leftRotate(msg, 4*i+32*4*j, msg_tmp);
            eval.add(msg_tmp0, msg_tmp, msg_tmp0);
        }
    }
    eval.mult(msg_tmp0, uni, msg_tmp0);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                eval.leftRotate(msg_tmp0, i+32*j+32*32*k, msg_tmp);
                eval.add(msg_out, msg_tmp, msg_out);
            }
        }
    }
    cout << "target value: " << endl;
    cout << msg_out[0] << endl;

    








    
    


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
