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
#include "ReLUbundle80.hpp"
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


    ///////////// Message ///////////////////


    cout << "Image Loading ..." << "\n";
    vector<vector<Ciphertext>> imageVec;

    for (int i = 1; i < 17; ++i) { // 313

        string str = "/app/HEAAN-ResNet/image/image_" + to_string(i) + string(".txt");
        vector<double> temp;
        txtreader(temp, str);
        vector<Ciphertext> out;
        imageCompiler(context, pack, enc, 5, temp, out);

        temp.clear();
        temp.shrink_to_fit();
        imageVec.push_back(out);

    }

    cout << "DONE, test for image encode ..." << "\n";

    Message dmsg;
    dec.decrypt(imageVec[0][0], sk, dmsg);
    printMessage(dmsg);

    cout << "DONE" << "\n";

    /*
    string str = "./image/image_" + to_string(313) + ".txt";
    vector<double> temp;
    txtreader(temp, str);
    for (int i = 0; i < 49152; ++i) temp.push_back(0);

    vector<Ciphertext> out;
    imageCompiler(context, pack, enc, temp, out);
    imageVec.push_back(out);
    */

    Plaintext ptxt_init(context); // for initializing
    Ciphertext ctxt_init(context); //추가
    double cnst = (double)(1.0 /40.0);

    // 1st conv
    vector<double> temp0;
    vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3(16, vector<vector<Plaintext>>(3, vector<Plaintext>(9, ptxt_init)));
    string path0 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block0conv0multiplicands16_3_3_3");
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

    cout << "test for conv1 multiplicands..." << "\n";
    printMessage(ecd.decode(block0conv0multiplicands16_3_3_3[0][0][0]));

    cout << "test for conv1 summands..." << "\n";
    printMessage(ecd.decode(block0conv0summands16[0]));


    // Convolution 0
    cout << "block0conv0 ..." << endl;
    timer.start(" block0conv0 ");
    vector<vector<Ciphertext>> ctxt_block0conv0_out(16, vector<Ciphertext>(16, ctxt_init));
    
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        #pragma omp parallel num_threads(5)
        {
        ctxt_block0conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 3, 16, imageVec[i], block0conv0multiplicands16_3_3_3);
        }
    }
    addBNsummands(context, eval, ctxt_block0conv0_out, block0conv0summands16, 16, 16);
    timer.end();

    imageVec.clear();
    imageVec.shrink_to_fit();

    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block0conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    //memory delete
    block0conv0multiplicands16_3_3_3.clear();
    block0conv0multiplicands16_3_3_3.shrink_to_fit();
    block0conv0summands16.clear();
    block0conv0summands16.shrink_to_fit();


    // // AppReLU
    cout << "block0relu0 ..." << endl;
    timer.start(" block0relu0 ");
    vector<vector<Ciphertext>> ctxt_block0relu0_out(16, vector<Ciphertext>(16, ctxt_init)); //초기화부분 추가
    ApproxReLU_bundle80(context,pack,eval,ctxt_block0conv0_out,ctxt_block0relu0_out);
    /*for (int i = 0; i < 16; ++i) {
        cout << "block0relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 16; ++ch) {
            ApproxReLU(context, eval, ctxt_block0conv0_out[i][ch], ctxt_block0relu0_out[i][ch]);
            eval.levelDown(ctxt_block0relu0_out[i][ch], 5, ctxt_block0relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block0relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "block0 DONE!" << "\n";






    ////////////////////
    /////// RB1 ////////
    ////////////////////

    // RB 1 - 1
    vector<double> temp1;
    vector<vector<vector<Plaintext>>> block1conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path1 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block1conv0multiplicands16_16_3_3");
    txtreader(temp1, path1);
    kernel_ptxt(context, temp1, block1conv0multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp1.clear();
    temp1.shrink_to_fit();

    vector<Plaintext> block1conv0summands16;
    vector<double> temp1a;
    string path1a = "/app/HEAAN-ResNet/kernel/summands/" + string("block1conv0summands16");
    Scaletxtreader(temp1a, path1a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp1a[i]);
        block1conv0summands16.push_back(ecd.encode(msg, 4, 0));
    }
    temp1a.clear();
    temp1a.shrink_to_fit();

    ///////////////////////// Main flow /////////////////////////////////////////
    cout << "block1conv0 ..." << endl;
    cout << "level of ctxt is " << ctxt_block0relu0_out[0][0].getLevel() << "\n";
    timer.start(" block1conv0 ");
    vector<vector<Ciphertext>> ctxt_block1conv0_out(16, vector<Ciphertext>(16, ctxt_init));
    
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(80)
        {
        ctxt_block1conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_block0relu0_out[i], block1conv0multiplicands16_16_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block1conv0_out, block1conv0summands16, 16, 16);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block1conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block1conv0multiplicands16_16_3_3.clear();
    block1conv0multiplicands16_16_3_3.shrink_to_fit();
    block1conv0summands16.clear();
    block1conv0summands16.shrink_to_fit();


    // AppReLU
    cout << "block1relu0 ..." << endl;
    timer.start(" block1relu0 ");
    vector<vector<Ciphertext>> ctxt_block1relu0_out(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block1conv0_out,ctxt_block1relu0_out);
    /*for (int i = 0; i < 16; ++i) {
        cout << "block1relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 16; ++ch) {
            ApproxReLU(context, eval, ctxt_block1conv0_out[i][ch], ctxt_block1relu0_out[i][ch]);
            eval.levelDown(ctxt_block1relu0_out[i][ch], 5, ctxt_block1relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block1relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "DONE!" << "\n";

    ctxt_block1conv0_out.clear();
    ctxt_block1conv0_out.shrink_to_fit();

    // RB 1 - 2
    vector<double> temp2;
    vector<vector<vector<Plaintext>>> block1conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path2 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block1conv1multiplicands16_16_3_3");
    txtreader(temp2, path2);
    kernel_ptxt(context, temp2, block1conv1multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp2.clear();
    temp2.shrink_to_fit();


    vector<Plaintext> block1conv1summands16;
    vector<double> temp2a;
    string path2a = "/app/HEAAN-ResNet/kernel/summands/" + string("block1conv1summands16");
    Scaletxtreader(temp2a, path2a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp2a[i]);
        block1conv1summands16.push_back(ecd.encode(msg, 4, 0));
    }

    temp2a.clear();
    temp2a.shrink_to_fit();

    // Second convolution
    cout << "block1conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block1relu0_out[0][0].getLevel() << "\n";
    timer.start(" block1conv1 ");
    vector<vector<Ciphertext>> ctxt_block1conv1_out(16, vector<Ciphertext>(16, ctxt_init));
    
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        ctxt_block1conv1_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_block1relu0_out[i], block1conv1multiplicands16_16_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block1conv1_out, block1conv1summands16, 16, 16);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block1conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block1conv1multiplicands16_16_3_3.clear();
    block1conv1multiplicands16_16_3_3.shrink_to_fit();
    block1conv1summands16.clear();
    block1conv1summands16.shrink_to_fit();




    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "block1add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block1add_out(16, vector<Ciphertext>(16, ctxt_init));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 16; ++i) {
        for (int ch = 0; ch < 16; ++ch) {
            eval.add(ctxt_block1conv1_out[i][ch], ctxt_block0relu0_out[i][ch], ctxt_block1add_out[i][ch]);
        }
    }
    cout << "DONE!" << "\n";
    ctxt_block0relu0_out.clear();
    ctxt_block0relu0_out.shrink_to_fit();
    ctxt_block1conv1_out.clear();
    ctxt_block1conv1_out.shrink_to_fit();


    // Last AppReLU
    cout << "block1relu1 ..." << endl;
    timer.start(" block1relu1 ");
    vector<vector<Ciphertext>> ctxt_block1relu1_out(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block1add_out,ctxt_block1relu1_out);
    /*for (int i = 0; i < 16; ++i) {
        cout << "block1relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 16; ++ch) {
            ApproxReLU(context, eval, ctxt_block1add_out[i][ch], ctxt_block1relu1_out[i][ch]);
            eval.levelDown(ctxt_block1relu1_out[i][ch], 5, ctxt_block1relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block1relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block1add_out.clear();
    ctxt_block1add_out.shrink_to_fit();
    cout << "RB1 DONE! " << "\n";






    ////////////////////
    /////// RB2 ////////
    ////////////////////


    // RB 2 - 1
    vector<double> temp3;
    vector<vector<vector<Plaintext>>> block2conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path3 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block2conv0multiplicands16_16_3_3");
    txtreader(temp3, path3);
    kernel_ptxt(context, temp3, block2conv0multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp3.clear();
    temp3.shrink_to_fit();


    vector<Plaintext> block2conv0summands16;
    vector<double> temp3a;
    string path3a = "/app/HEAAN-ResNet/kernel/summands/" + string("block2conv0summands16");
    Scaletxtreader(temp3a, path3a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp3a[i]);
        block2conv0summands16.push_back(ecd.encode(msg, 4, 0));
    }
    temp3a.clear();
    temp3a.shrink_to_fit();

    ///////////////////////// Main flow /////////////////////////////////////////
    cout << "block2conv0 ..." << endl;
    cout << "level of ctxt is " << ctxt_block1relu0_out[0][0].getLevel() << "\n";
    timer.start(" block2conv0 ");
    vector<vector<Ciphertext>> ctxt_block2conv0_out(16, vector<Ciphertext>(16, ctxt_init));
                               
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) { 
        #pragma omp parallel num_threads(5)
        {
        ctxt_block2conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_block1relu1_out[i], block2conv0multiplicands16_16_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block2conv0_out, block2conv0summands16, 16, 16);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block2conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block2conv0multiplicands16_16_3_3.clear();
    block2conv0multiplicands16_16_3_3.shrink_to_fit();
    block2conv0summands16.clear();
    block2conv0summands16.shrink_to_fit();


    // AppReLU
    cout << "block2relu0 ..." << endl;
    timer.start(" block2relu0 ");
    vector<vector<Ciphertext>> ctxt_block2relu0_out(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block2conv0_out,ctxt_block2relu0_out);
    /*for (int i = 0; i < 16; ++i) {
        cout << "block2relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 16; ++ch) {
            ApproxReLU(context, eval, ctxt_block2conv0_out[i][ch], ctxt_block2relu0_out[i][ch]);
            eval.levelDown(ctxt_block2relu0_out[i][ch], 5, ctxt_block2relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block2relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "DONE!" << "\n";

    ctxt_block2conv0_out.clear();
    ctxt_block2conv0_out.shrink_to_fit();


    // RB 2 - 2
    vector<double> temp4;
    vector<vector<vector<Plaintext>>> block2conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path4 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block2conv1multiplicands16_16_3_3");
    txtreader(temp4, path4);
    kernel_ptxt(context, temp4, block2conv1multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp4.clear();
    temp4.shrink_to_fit();


    vector<Plaintext> block2conv1summands16;
    vector<double> temp4a;
    string path4a = "/app/HEAAN-ResNet/kernel/summands/" + string("block2conv1summands16");
    Scaletxtreader(temp4a, path4a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp4a[i]);
        block2conv1summands16.push_back(ecd.encode(msg, 4, 0));
    }
    temp4a.clear();
    temp4a.shrink_to_fit();

    // Second convolution
    cout << "block2conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block2relu0_out[0][0].getLevel() << "\n";
    timer.start(" block2conv1 ");
    vector<vector<Ciphertext>> ctxt_block2conv1_out(16, vector<Ciphertext>(16, ctxt_init));
                               
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        ctxt_block2conv1_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_block2relu0_out[i], block2conv1multiplicands16_16_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block2conv1_out, block2conv1summands16, 16, 16);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block2conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block2conv1multiplicands16_16_3_3.clear();
    block2conv1multiplicands16_16_3_3.shrink_to_fit();
    block2conv1summands16.clear();
    block2conv1summands16.shrink_to_fit();




    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "block2add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block2add_out(16, vector<Ciphertext>(16, ctxt_init));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 16; ++i) {
        for (int ch = 0; ch < 16; ++ch) {
            eval.add(ctxt_block2conv1_out[i][ch], ctxt_block2relu0_out[i][ch], ctxt_block2add_out[i][ch]);
        }
    }
    cout << "DONE!" << "\n";
    ctxt_block2relu0_out.clear();
    ctxt_block2relu0_out.shrink_to_fit();
    ctxt_block2conv1_out.clear();
    ctxt_block2conv1_out.shrink_to_fit();


    // Last AppReLU
    cout << "block2relu1 ..." << endl;
    timer.start(" block2relu1 ");
    vector<vector<Ciphertext>> ctxt_block2relu1_out(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block2add_out,ctxt_block2relu1_out);
    /*for (int i = 0; i < 16; ++i) {
        cout << "block2relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 16; ++ch) {
            ApproxReLU(context, eval, ctxt_block2add_out[i][ch], ctxt_block2relu1_out[i][ch]);
            eval.levelDown(ctxt_block2relu1_out[i][ch], 5, ctxt_block2relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block2relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block2add_out.clear();
    ctxt_block2add_out.shrink_to_fit();
    cout << "RB2 DONE! " << "\n";



    ////////////////////
    /////// RB3 ////////
    ////////////////////


    levelDownBundle(context, pack, eval, ctxt_block2relu1_out, 5);

    // RB 3 - 1
    vector<double> temp5;
    vector<vector<vector<Plaintext>>> block3conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path5 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block3conv0multiplicands16_16_3_3");
    txtreader(temp5, path5);
    kernel_ptxt(context, temp5, block3conv0multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp5.clear();
    temp5.shrink_to_fit();


    vector<Plaintext> block3conv0summands16;
    vector<double> temp5a;
    string path5a = "/app/HEAAN-ResNet/kernel/summands/" + string("block3conv0summands16");
    Scaletxtreader(temp5a, path5a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp5a[i]);
        block3conv0summands16.push_back(ecd.encode(msg, 4, 0));
    }
    temp5a.clear();
    temp5a.shrink_to_fit();


    ///////////////////////// Main flow /////////////////////////////////////////
    cout << "block3conv0 ..." << endl;
    cout << "level of ctxt is " << ctxt_block2relu1_out[0][0].getLevel() << "\n";
    timer.start(" block3conv0 ");
    vector<vector<Ciphertext>> ctxt_block3conv0_out(16, vector<Ciphertext>(16, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        ctxt_block3conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_block2relu1_out[i], block3conv0multiplicands16_16_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block3conv0_out, block3conv0summands16, 16, 16);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block3conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block3conv0multiplicands16_16_3_3.clear();
    block3conv0multiplicands16_16_3_3.shrink_to_fit();
    block3conv0summands16.clear();
    block3conv0summands16.shrink_to_fit();


    // AppReLU
    cout << "block3relu0 ..." << endl;
    timer.start(" block3relu0 ");
    vector<vector<Ciphertext>> ctxt_block3relu0_out(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block3conv0_out,ctxt_block3relu0_out);
    /*for (int i = 0; i < 16; ++i) {
        cout << "block3relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 16; ++ch) {
            ApproxReLU(context, eval, ctxt_block3conv0_out[i][ch], ctxt_block3relu0_out[i][ch]);
            eval.levelDown(ctxt_block3relu0_out[i][ch], 5, ctxt_block3relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block3relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "DONE!" << "\n";

    ctxt_block3conv0_out.clear();
    ctxt_block3conv0_out.shrink_to_fit();

    // RB 3 - 2
    vector<double> temp6;
    vector<vector<vector<Plaintext>>> block3conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path6 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block3conv1multiplicands16_16_3_3");
    txtreader(temp6, path6);
    kernel_ptxt(context, temp6, block3conv1multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);
    temp6.clear();
    temp6.shrink_to_fit();


    vector<Plaintext> block3conv1summands16;
    vector<double> temp6a;
    string path6a = "/app/HEAAN-ResNet/kernel/summands/" + string("block3conv1summands16");
    Scaletxtreader(temp6a, path6a, cnst);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp6a[i]);
        block3conv1summands16.push_back(ecd.encode(msg, 4, 0));
    }
    temp6a.clear();
    temp6a.shrink_to_fit();


    // Second convolution
    cout << "block3conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block3relu0_out[0][0].getLevel() << "\n";
    timer.start(" block3conv1 ");
    vector<vector<Ciphertext>> ctxt_block3conv1_out(16, vector<Ciphertext>(16, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) {
        #pragma omp parallel num_threads(5)
        {
        ctxt_block3conv1_out[i] = Conv(context, pack, eval, 32, 1, 1, 16, 16, ctxt_block3relu0_out[i], block3conv1multiplicands16_16_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block3conv1_out, block3conv1summands16, 16, 16);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block3conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block3conv1multiplicands16_16_3_3.clear();
    block3conv1multiplicands16_16_3_3.shrink_to_fit();
    block3conv1summands16.clear();
    block3conv1summands16.shrink_to_fit();




    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "block3add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block3add_out(16, vector<Ciphertext>(16, ctxt_init));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 16; ++i) {
        for (int ch = 0; ch < 16; ++ch) {
            eval.add(ctxt_block3conv1_out[i][ch], ctxt_block3relu0_out[i][ch], ctxt_block3add_out[i][ch]);
        }
    }
    cout << "DONE!" << "\n";
    ctxt_block3relu0_out.clear();
    ctxt_block3relu0_out.shrink_to_fit();
    ctxt_block3conv1_out.clear();
    ctxt_block3conv1_out.shrink_to_fit();


    // Last AppReLU
    cout << "block3relu1 ..." << endl;
    timer.start(" block3relu1 ");
    vector<vector<Ciphertext>> ctxt_block3relu1_out(16, vector<Ciphertext>(16, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block3add_out,ctxt_block3relu1_out);
    /*for (int i = 0; i < 16; ++i) {
        cout << "block3relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 16; ++ch) {
            ApproxReLU(context, eval, ctxt_block3add_out[i][ch], ctxt_block3relu1_out[i][ch]);
            eval.levelDown(ctxt_block3relu1_out[i][ch], 5, ctxt_block3relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block3relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block3add_out.clear();
    ctxt_block3add_out.shrink_to_fit();
    cout << "RB3 DONE! " << "\n";




    // Down Sampling (Residual) Block 1




    //////////////////////////////
    ///////// DSB 1//////////////
    /////////////////////////////
    cout << "DSB 1 ..." << endl;

    ///////////////////// Residual flow ////////////////////////////
    // Convolution

    // DSB 1 - res
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

    cout << "block4conv_onebyone ..." << endl;
    cout << "level of ctxt is " << ctxt_block3relu1_out[0][0].getLevel() << "\n";
    timer.start(" block4conv_onebyone .. ");
    vector<vector<Ciphertext>> ctxt_block4conv_onebyone_out(16, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        #pragma omp parallel num_threads(5)
        {
        ctxt_block4conv_onebyone_out[i] = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_block3relu1_out[i], block4conv_onebyone_multiplicands32_16_1_1);
        }
    }

    block4conv_onebyone_multiplicands32_16_1_1.clear();
    block4conv_onebyone_multiplicands32_16_1_1.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block4conv_onebyone_out[0][0].getLevel() << "\n";
    cout << "and decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block4conv_onebyone_out[0][0], sk, dmsg);
    printMessage(dmsg);

    // MPP input bundle making
    cout << "block4MPP1 and BN summand ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block4MPP1_in(3, vector<vector<Ciphertext>>(32, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block4MPP1_in[i][ch][k] = ctxt_block4conv_onebyone_out[i + k][ch];
            }
        }
    }
    ctxt_block4conv_onebyone_out.clear();
    ctxt_block4conv_onebyone_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block4MPP1_out(4, vector<Ciphertext>(32, ctxt_init));

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            ctxt_block4MPP1_out[i][ch] = MPPacking(context, pack, eval, 32, ctxt_block4MPP1_in[i][ch]);
        }
    }

    ctxt_block4MPP1_in.clear();
    ctxt_block4MPP1_in.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block4MPP1_out, block4conv_onebyone_summands32, 4, 32);
    timer.end();

    block4conv_onebyone_summands32.clear();
    block4conv_onebyone_summands32.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block4MPP1_out[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block4MPP1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    ///////////////////////// Main flow /////////////////////////////////////////


    // DSB 1 - 1
    vector<double> temp8;
    vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3(32, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path8 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block4conv0multiplicands32_16_3_3");
    txtreader(temp8, path8);
    kernel_ptxt(context, temp8, block4conv0multiplicands32_16_3_3, 5, 1, 2, 32, 16, 3, ecd);
    temp8.clear();
    temp8.shrink_to_fit();


    vector<Plaintext> block4conv0summands32;
    vector<double> temp8a;
    string path8a = "/app/HEAAN-ResNet/kernel/summands/" + string("block4conv0summands32");
    Scaletxtreader(temp8a, path8a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp8a[i]);
        block4conv0summands32.push_back(ecd.encode(msg, 4, 0));
    }
    temp8a.clear();
    temp8a.shrink_to_fit();

    cout << "block4conv0 ..." << endl;
    timer.start(" block4conv0 ");
    vector<vector<Ciphertext>> ctxt_block4conv0_out(16, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 16; ++i) { // 서로 다른 img
        #pragma omp parallel num_threads(5)
        {
        ctxt_block4conv0_out[i] = Conv(context, pack, eval, 32, 1, 2, 16, 32, ctxt_block3relu1_out[i], block4conv0multiplicands32_16_3_3);
        }
    }

    ctxt_block3relu1_out.clear();
    ctxt_block3relu1_out.shrink_to_fit();

    block4conv0multiplicands32_16_3_3.clear();
    block4conv0multiplicands32_16_3_3.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block4conv0_out[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block4conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    // MPP input bundle making
    cout << "block4MPP0 and BN summand ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block4MPP0_in(4, vector<vector<Ciphertext>>(32, vector<Ciphertext>(4, ctxt_init)));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block4MPP0_in[i][ch][k] = ctxt_block4conv0_out[i + k][ch];
            }
        }
    }

    ctxt_block4conv0_out.clear();
    ctxt_block4conv0_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block4MPP0_out(4, vector<Ciphertext>(32, ctxt_init));

    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            ctxt_block4MPP0_out[i][ch] = MPPacking(context, pack, eval, 32, ctxt_block4MPP0_in[i][ch]);
        }
    }

    ctxt_block4MPP0_in.clear();
    ctxt_block4MPP0_in.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block4MPP0_out, block4conv0summands32, 4, 32);
    timer.end();

    block4conv0summands32.clear();
    block4conv0summands32.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block4MPP0_out[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block4MPP0_out[0][0], sk, dmsg);
    printMessage(dmsg);

    // ctxt_block4MPP0_out 첫번째 : 서로 다른 img, 두번째 : ch.

    // AppReLU
    cout << "block4relu0 ..." << endl;
    timer.start(" block4relu0 ");
    vector<vector<Ciphertext>> ctxt_block4relu0_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block4MPP0_out,ctxt_block4relu0_out);
    /*for (int i = 0; i < 4; ++i) {
        cout << "block4relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 32; ++ch) {
            ApproxReLU(context, eval, ctxt_block4MPP0_out[i][ch], ctxt_block4relu0_out[i][ch]);
            eval.levelDown(ctxt_block4relu0_out[i][ch], 5, ctxt_block4relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block4relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block4MPP0_out.clear();
    ctxt_block4MPP0_out.shrink_to_fit();

    cout << "DONE!" << "\n";

    // Second convolution

    // DSB 1 - 2
    vector<double> temp9;
    vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path9 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block4conv1multiplicands32_32_3_3");
    txtreader(temp9, path9);
    kernel_ptxt(context, temp9, block4conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp9.clear();
    temp9.shrink_to_fit();


    vector<Plaintext> block4conv1summands32;
    vector<double> temp9a;
    string path9a = "/app/HEAAN-ResNet/kernel/summands/" + string("block4conv1summands32");
    Scaletxtreader(temp9a, path9a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp9a[i]);
        block4conv1summands32.push_back(ecd.encode(msg, 4, 0));
    }
    temp9a.clear();
    temp9a.shrink_to_fit();

    cout << "block4conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block4relu0_out[0][0].getLevel() << "\n";
    timer.start(" block4conv1 ");
    vector<vector<Ciphertext>> ctxt_block4conv1_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(20)
        {
        ctxt_block4conv1_out[i] = Conv(context, pack, eval, 32, 1, 1, 32, 32, ctxt_block4relu0_out[i], block4conv1multiplicands32_32_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block4conv1_out, block4conv1summands32, 4, 32);
    timer.end();

    ctxt_block4relu0_out.clear();
    ctxt_block4relu0_out.shrink_to_fit();

    block4conv1multiplicands32_32_3_3.clear();
    block4conv1multiplicands32_32_3_3.shrink_to_fit();
    block4conv1summands32.clear();
    block4conv1summands32.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block4conv1_out[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block4conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);



    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Main flow + Residual flow ..." << endl;
    vector<vector<Ciphertext>> ctxt_block4add_out(4, vector<Ciphertext>(32, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            eval.add(ctxt_block4conv1_out[i][ch], ctxt_block4MPP1_out[i][ch], ctxt_block4add_out[i][ch]);
        }
    }

    ctxt_block4conv1_out.clear();
    ctxt_block4conv1_out.shrink_to_fit();
    ctxt_block4MPP1_out.clear();
    ctxt_block4MPP1_out.shrink_to_fit();

    cout << "DONE!" << "\n";

    // Last AppReLU
    cout << "block4relu1 ..." << endl;
    timer.start(" block4relu1 ");
    vector<vector<Ciphertext>> ctxt_block4relu1_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block4add_out,ctxt_block4relu1_out);
    /*for (int i = 0; i < 4; ++i) {
        cout << "block4relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 32; ++ch) {
            ApproxReLU(context, eval, ctxt_block4add_out[i][ch], ctxt_block4relu1_out[i][ch]);
            eval.levelDown(ctxt_block4relu1_out[i][ch], 5, ctxt_block4relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block4relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block4add_out.clear();
    ctxt_block4add_out.shrink_to_fit();
    cout << "DSB1 DONE!" << "\n";



    // Residual Block 4, 5



    ///////////////////////////////////
    ////////// RB 4 ////////////////
    //////////////////////////////



    levelDownBundle(context, pack, eval, ctxt_block4relu1_out, 5);

    ///////////////////////// Main flow /////////////////////////////////////////


        // RB 4 - 1
    vector<double> temp10;
    vector<vector<vector<Plaintext>>> block5conv0multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path10 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block5conv0multiplicands32_32_3_3");
    txtreader(temp10, path10);
    kernel_ptxt(context, temp10, block5conv0multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp10.clear();
    temp10.shrink_to_fit();


    vector<Plaintext> block5conv0summands32;
    vector<double> temp10a;
    string path10a = "/app/HEAAN-ResNet/kernel/summands/" + string("block5conv0summands32");
    Scaletxtreader(temp10a, path10a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp10a[i]);
        block5conv0summands32.push_back(ecd.encode(msg, 4, 0));
    }


    cout << "block5conv0 ..." << endl;
    cout << "level of ctxt is " << ctxt_block4relu1_out[0][0].getLevel() << "\n";
    timer.start(" block5conv0 ");
    vector<vector<Ciphertext>> ctxt_block5conv0_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 4; ++i) { // 서로 다른 img
        //cout << "i = " << i << endl;
        #pragma omp parallel num_threads(20)
        {
        ctxt_block5conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 32, 32, ctxt_block4relu1_out[i], block5conv0multiplicands32_32_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block5conv0_out, block5conv0summands32, 4, 32);
    timer.end();

    block5conv0multiplicands32_32_3_3.clear();
    block5conv0multiplicands32_32_3_3.shrink_to_fit();
    block5conv0summands32.clear();
    block5conv0summands32.shrink_to_fit();

    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block5conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);




    // AppReLU
    cout << "block5relu0 ..." << endl;
    timer.start(" block5relu0 ");
    vector<vector<Ciphertext>> ctxt_block5relu0_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block5conv0_out,ctxt_block5relu0_out);
    /*for (int i = 0; i < 4; ++i) {
        cout << "block5relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 32; ++ch) {
            ApproxReLU(context, eval, ctxt_block5conv0_out[i][ch], ctxt_block5relu0_out[i][ch]);
            eval.levelDown(ctxt_block5relu0_out[i][ch], 5, ctxt_block5relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block5relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "DONE!" << "\n";

    ctxt_block5conv0_out.clear();
    ctxt_block5conv0_out.shrink_to_fit();

    // Second convolution

    // RB 4 - 2
    vector<double> temp11;
    vector<vector<vector<Plaintext>>> block5conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path11 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block5conv1multiplicands32_32_3_3");
    txtreader(temp11, path11);
    kernel_ptxt(context, temp11, block5conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp11.clear();
    temp11.shrink_to_fit();


    vector<Plaintext> block5conv1summands32;
    vector<double> temp11a;
    string path11a = "/app/HEAAN-ResNet/kernel/summands/" + string("block5conv1summands32");
    Scaletxtreader(temp11a, path11a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp11a[i]);
        block5conv1summands32.push_back(ecd.encode(msg, 4, 0));
    }
    temp11a.clear();
    temp11a.shrink_to_fit();


    cout << "block5conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block5relu0_out[0][0].getLevel() << "\n";
    timer.start(" block5conv1 ");
    vector<vector<Ciphertext>> ctxt_block5conv1_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(20)
        {
        ctxt_block5conv1_out[i] = Conv(context, pack, eval, 32, 1, 1, 32, 32, ctxt_block5relu0_out[i], block5conv1multiplicands32_32_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block5conv1_out, block5conv1summands32, 4, 32);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block5conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block5conv1multiplicands32_32_3_3.clear();
    block5conv1multiplicands32_32_3_3.shrink_to_fit();
    block5conv1summands32.clear();
    block5conv1summands32.shrink_to_fit();




    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "block5add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block5add_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            eval.add(ctxt_block5conv1_out[i][ch], ctxt_block4relu1_out[i][ch], ctxt_block5add_out[i][ch]);
        }
    }
    cout << "DONE!" << "\n";
    ctxt_block4relu1_out.clear();
    ctxt_block4relu1_out.shrink_to_fit();
    ctxt_block5conv1_out.clear();
    ctxt_block5conv1_out.shrink_to_fit();


    // Last AppReLU
    cout << "block5relu1 ..." << endl;
    timer.start(" block5relu1 ");
    vector<vector<Ciphertext>> ctxt_block5relu1_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block5add_out,ctxt_block5relu1_out);
    /*for (int i = 0; i < 4; ++i) {
        cout << "block5relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 32; ++ch) {
            ApproxReLU(context, eval, ctxt_block5add_out[i][ch], ctxt_block5relu1_out[i][ch]);
            eval.levelDown(ctxt_block5relu1_out[i][ch], 5, ctxt_block5relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block5relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block5add_out.clear();
    ctxt_block5add_out.shrink_to_fit();
    cout << "RB4 DONE! " << "\n";







    ////////////////////////////
    ////// RB 5 ////////////
    ///////////////////////

    ///////////////////////// Main flow /////////////////////////////////////////


    // RB 5 - 1
    vector<double> temp12;
    vector<vector<vector<Plaintext>>> block6conv0multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path12 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block6conv0multiplicands32_32_3_3");
    txtreader(temp12, path12);
    kernel_ptxt(context, temp12, block6conv0multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp12.clear();
    temp12.shrink_to_fit();

    vector<Plaintext> block6conv0summands32;
    vector<double> temp12a;
    string path12a = "/app/HEAAN-ResNet/kernel/summands/" + string("block6conv0summands32");
    Scaletxtreader(temp12a, path12a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp12a[i]);
        block6conv0summands32.push_back(ecd.encode(msg, 4, 0));
    }
    temp12a.clear();
    temp12a.shrink_to_fit();



    cout << "block6conv0 ..." << endl;
    cout << "level of ctxt is " << ctxt_block5relu1_out[0][0].getLevel() << "\n";
    timer.start(" block6conv0 ");
    vector<vector<Ciphertext>> ctxt_block6conv0_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 4; ++i) { // 서로 다른 img
        //cout << "i = " << i << endl;
        #pragma omp parallel num_threads(20)
        {
        ctxt_block6conv0_out[i] = Conv(context, pack, eval, 32, 1, 1, 32, 32, ctxt_block5relu1_out[i], block6conv0multiplicands32_32_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block6conv0_out, block6conv0summands32, 4, 32);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block6conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block6conv0multiplicands32_32_3_3.clear();
    block6conv0multiplicands32_32_3_3.shrink_to_fit();
    block6conv0summands32.clear();
    block6conv0summands32.shrink_to_fit();


    // AppReLU
    cout << "block6relu0 ..." << endl;
    timer.start(" block6relu0 ");
    vector<vector<Ciphertext>> ctxt_block6relu0_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block6conv0_out,ctxt_block6relu0_out);
    /*for (int i = 0; i < 4; ++i) {
        cout << "block6relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 32; ++ch) {
            ApproxReLU(context, eval, ctxt_block6conv0_out[i][ch], ctxt_block6relu0_out[i][ch]);
            eval.levelDown(ctxt_block6relu0_out[i][ch], 5, ctxt_block6relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block6relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "DONE!" << "\n";

    ctxt_block6conv0_out.clear();
    ctxt_block6conv0_out.shrink_to_fit();

    // Second convolution

    // RB 5 - 2
    vector<double> temp13;
    vector<vector<vector<Plaintext>>> block6conv1multiplicands32_32_3_3(32, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path13 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block6conv1multiplicands32_32_3_3");
    txtreader(temp13, path13);
    kernel_ptxt(context, temp13, block6conv1multiplicands32_32_3_3, 5, 2, 1, 32, 32, 3, ecd);
    temp13.clear();
    temp13.shrink_to_fit();


    vector<Plaintext> block6conv1summands32;
    vector<double> temp13a;
    string path13a = "/app/HEAAN-ResNet/kernel/summands/" + string("block6conv1summands32");
    Scaletxtreader(temp13a, path13a, cnst);

    for (int i = 0; i < 32; ++i) {
        Message msg(log_slots, temp13a[i]);
        block6conv1summands32.push_back(ecd.encode(msg, 4, 0));
    }
    temp13a.clear();
    temp13a.shrink_to_fit();



    cout << "block6conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block6relu0_out[0][0].getLevel() << "\n";
    timer.start(" block6conv1 ");
    vector<vector<Ciphertext>> ctxt_block6conv1_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 4; ++i) {
        #pragma omp parallel num_threads(20)
        {
        ctxt_block6conv1_out[i] = Conv(context, pack, eval, 32, 1, 1, 32, 32, ctxt_block6relu0_out[i], block6conv1multiplicands32_32_3_3);
        }
    }

    addBNsummands(context, eval, ctxt_block6conv1_out, block6conv1summands32, 4, 32);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block6conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block6conv1multiplicands32_32_3_3.clear();
    block6conv1multiplicands32_32_3_3.shrink_to_fit();
    block6conv1summands32.clear();
    block6conv1summands32.shrink_to_fit();




    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "block6add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block6add_out(4, vector<Ciphertext>(32, ctxt_init));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 4; ++i) {
        for (int ch = 0; ch < 32; ++ch) {
            eval.add(ctxt_block6conv1_out[i][ch], ctxt_block5relu1_out[i][ch], ctxt_block6add_out[i][ch]);
        }
    }
    cout << "DONE!" << "\n";
    ctxt_block5relu1_out.clear();
    ctxt_block5relu1_out.shrink_to_fit();
    ctxt_block6conv1_out.clear();
    ctxt_block6conv1_out.shrink_to_fit();


    // Last AppReLU
    cout << "block6relu1 ..." << endl;
    timer.start(" block6relu1 ");
    vector<vector<Ciphertext>> ctxt_block6relu1_out(4, vector<Ciphertext>(32, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block6add_out,ctxt_block6relu1_out);
    /*for (int i = 0; i < 4; ++i) {
        cout << "block6relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 32; ++ch) {
            ApproxReLU(context, eval, ctxt_block6add_out[i][ch], ctxt_block6relu1_out[i][ch]);
            eval.levelDown(ctxt_block6relu1_out[i][ch], 5, ctxt_block6relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block6relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block6add_out.clear();
    ctxt_block6add_out.shrink_to_fit();
    cout << "RB5 DONE! " << "\n";



    // Down Sampling (Residual) Block 2
    // DSB 2



    ////////////////////////
    /////// DSB 2 //////////
    /////////////////////////


    ///////////////////// Residual flow ////////////////////////////
    // Convolution


    vector<double> temp14;
    vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path14 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block7conv_onebyone_multiplicands64_32_1_1");
    txtreader(temp14, path14);
    kernel_ptxt(context, temp14, block7conv_onebyone_multiplicands64_32_1_1, 5, 2, 2, 64, 32, 1, ecd);
    temp14.clear();
    temp14.shrink_to_fit();

    vector<Plaintext> block7conv_onebyone_summands64;
    vector<double> temp14a;
    string path14a = "/app/HEAAN-ResNet/kernel/summands/" + string("block7conv_onebyone_summands64");
    Scaletxtreader(temp14a, path14a, cnst);


    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp14a[i]);
        block7conv_onebyone_summands64.push_back(ecd.encode(msg, 4, 0));
    }
    temp14a.clear();
    temp14a.shrink_to_fit();


    cout << "block7conv_onebyone ..." << endl;
    timer.start(" block7conv_onebyone .. ");
    cout << "level of ctxt is " << ctxt_block6relu1_out[0][0].getLevel() << "\n";
    vector<vector<Ciphertext>> ctxt_block7conv_onebyone_out(4, vector<Ciphertext>(64, ctxt_init));
    #pragma omp parallel for num_threads(80)
    for (int i = 0; i < 4; ++i) { // 서로 다른 img
        #pragma omp parallel num_threads(20)
        {
        ctxt_block7conv_onebyone_out[i] = Conv(context, pack, eval, 32, 1, 2, 32, 64, ctxt_block6relu1_out[i], block7conv_onebyone_multiplicands64_32_1_1);
        }
    }


    block7conv_onebyone_multiplicands64_32_1_1.clear();
    block7conv_onebyone_multiplicands64_32_1_1.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block7conv_onebyone_out[0][0].getLevel() << "\n";
    cout << "and decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block7conv_onebyone_out[0][0], sk, dmsg);
    printMessage(dmsg);

    // MPP input bundle making
    cout << "block7MPP1 and BN summand ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block7MPP1_in(1, vector<vector<Ciphertext>>(64, vector<Ciphertext>(4, ctxt_init)));

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block7MPP1_in[i][ch][k] = ctxt_block7conv_onebyone_out[i + k][ch];
            }
        }
    }
    ctxt_block7conv_onebyone_out.clear();
    ctxt_block7conv_onebyone_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block7MPP1_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            ctxt_block7MPP1_out[i][ch] = MPPacking(context, pack, eval, 32, ctxt_block7MPP1_in[i][ch]);
        }
    }

    ctxt_block7MPP1_in.clear();
    ctxt_block7MPP1_in.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block7MPP1_out, block7conv_onebyone_summands64, 1, 64);
    timer.end();

    block7conv_onebyone_summands64.clear();
    block7conv_onebyone_summands64.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block7MPP1_out[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block7MPP1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    ///////////////////////// Main flow /////////////////////////////////////////

    vector<double> temp15;
    vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3(64, vector<vector<Plaintext>>(32, vector<Plaintext>(9, ptxt_init)));
    string path15 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block7conv0multiplicands64_32_3_3");
    txtreader(temp15, path15);
    kernel_ptxt(context, temp15, block7conv0multiplicands64_32_3_3, 5, 2, 2, 64, 32, 3, ecd);
    temp15.clear();
    temp15.shrink_to_fit();


    vector<Plaintext> block7conv0summands64;
    vector<double> temp15a;
    string path15a = "/app/HEAAN-ResNet/kernel/summands/" + string("block7conv0summands64");
    Scaletxtreader(temp15a, path15a, cnst);

    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp15a[i]);
        block7conv0summands64.push_back(ecd.encode(msg, 4, 0));
    }
    temp15a.clear();
    temp15a.shrink_to_fit();

    cout << "block7conv0 ..." << endl;
    timer.start(" block7conv0 ");
    vector<vector<Ciphertext>> ctxt_block7conv0_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) { // 서로 다른 img
        ctxt_block7conv0_out[i] = Conv_parallel(context, pack, eval, 32, 1, 2, 32, 64, ctxt_block6relu1_out[i], block7conv0multiplicands64_32_3_3);
    }

    ctxt_block6relu1_out.clear();
    ctxt_block6relu1_out.shrink_to_fit();

    block7conv0multiplicands64_32_3_3.clear();
    block7conv0multiplicands64_32_3_3.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block7conv0_out[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block7conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    // MPP input bundle making
    cout << "block7MPP0 and BN summand ..." << endl;
    vector<vector<vector<Ciphertext>>> ctxt_block7MPP0_in(1, vector<vector<Ciphertext>>(64, vector<Ciphertext>(4, ctxt_init)));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            for (int k = 0; k < 4; ++k) {
                ctxt_block7MPP0_in[i][ch][k] = ctxt_block7conv0_out[i + k][ch];
            }
        }
    }

    ctxt_block7conv0_out.clear();
    ctxt_block7conv0_out.shrink_to_fit();

    // MPP
    vector<vector<Ciphertext>> ctxt_block7MPP0_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            ctxt_block7MPP0_out[i][ch] = MPPacking(context, pack, eval, 32, ctxt_block7MPP0_in[i][ch]);
        }
    }

    ctxt_block7MPP0_in.clear();
    ctxt_block7MPP0_in.shrink_to_fit();

    addBNsummands(context, eval, ctxt_block7MPP0_out, block7conv0summands64, 1, 64);
    timer.end();

    block7conv0summands64.clear();
    block7conv0summands64.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block7MPP0_out[0][0].getLevel() << "\n";
    cout << "and decrypted messagee is ... " << "\n";
    dec.decrypt(ctxt_block7MPP0_out[0][0], sk, dmsg);
    printMessage(dmsg);

    // ctxt_block7MPP0_out 첫번째 : 서로 다른 img, 두번째 : ch.

    // AppReLU
    cout << "block7relu0 ..." << endl;
    timer.start(" block7relu0 ");
    vector<vector<Ciphertext>> ctxt_block7relu0_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block7MPP0_out,ctxt_block7relu0_out);
    /*for (int i = 0; i < 1; ++i) {
        cout << "block7relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 64; ++ch) {
            ApproxReLU(context, eval, ctxt_block7MPP0_out[i][ch], ctxt_block7relu0_out[i][ch]);
            eval.levelDown(ctxt_block7relu0_out[i][ch], 5, ctxt_block7relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block7relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block7MPP0_out.clear();
    ctxt_block7MPP0_out.shrink_to_fit();

    cout << "DONE!" << "\n";

    // Second convolution

    vector<double> temp16;
    vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path16 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block7conv1multiplicands64_64_3_3");
    txtreader(temp16, path16);
    kernel_ptxt(context, temp16, block7conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    temp16.clear();
    temp16.shrink_to_fit();

    vector<Plaintext> block7conv1summands64;
    vector<double> temp16a;
    string path16a = "/app/HEAAN-ResNet/kernel/summands/" + string("block7conv1summands64");
    Scaletxtreader(temp16a, path16a, cnst);

    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp16a[i]);
        block7conv1summands64.push_back(ecd.encode(msg, 4, 0));
    }

    temp16a.clear();
    temp16a.shrink_to_fit();


    cout << "block7conv1 ..." << endl;
    timer.start(" block7conv1 ");
    vector<vector<Ciphertext>> ctxt_block7conv1_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) {
        ctxt_block7conv1_out[i] = Conv_parallel(context, pack, eval, 32, 1, 1, 64, 64, ctxt_block7relu0_out[i], block7conv1multiplicands64_64_3_3);
    }

    addBNsummands(context, eval, ctxt_block7conv1_out, block7conv1summands64, 1, 64);
    timer.end();

    ctxt_block7relu0_out.clear();
    ctxt_block7relu0_out.shrink_to_fit();

    block7conv1multiplicands64_64_3_3.clear();
    block7conv1multiplicands64_64_3_3.shrink_to_fit();
    block7conv1summands64.clear();
    block7conv1summands64.shrink_to_fit();

    cout << "Done!! level of ctxt is " << ctxt_block7conv1_out[0][0].getLevel() << "\n";
    cout << "and decrypted message is ... " << "\n";
    dec.decrypt(ctxt_block7conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);



    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "Main flow + Residual flow ..." << endl;
    vector<vector<Ciphertext>> ctxt_block7add_out(1, vector<Ciphertext>(64, ctxt_init));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            eval.add(ctxt_block7conv1_out[i][ch], ctxt_block7MPP1_out[i][ch], ctxt_block7add_out[i][ch]);
        }
    }

    ctxt_block7conv1_out.clear();
    ctxt_block7conv1_out.shrink_to_fit();
    ctxt_block7MPP1_out.clear();
    ctxt_block7MPP1_out.shrink_to_fit();

    cout << "DONE!" << "\n";

    // Last AppReLU
    cout << "block7relu1 ..." << endl;
    timer.start(" block7relu1 ");
    vector<vector<Ciphertext>> ctxt_block7relu1_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block7add_out,ctxt_block7relu1_out);
    /*for (int i = 0; i < 1; ++i) {
        cout << "block7relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 64; ++ch) {
            ApproxReLU(context, eval, ctxt_block7add_out[i][ch], ctxt_block7relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block7relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block7add_out.clear();
    ctxt_block7add_out.shrink_to_fit();
    cout << "DSB2 DONE!" << "\n";



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

    vector<Plaintext> block8conv0summands64;
    vector<double> temp17a;
    string path17a = "/app/HEAAN-ResNet/kernel/summands/" + string("block8conv0summands64");
    Scaletxtreader(temp17a, path17a, cnst);

    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp17a[i]);
        block8conv0summands64.push_back(ecd.encode(msg, 4, 0));
    }
    temp17a.clear();
    temp17a.shrink_to_fit();



    cout << "block8conv0 ..." << endl;
    cout << "level of ctxt is " << ctxt_block7relu1_out[0][0].getLevel() << "\n";
    timer.start(" block8conv0 ");
    vector<vector<Ciphertext>> ctxt_block8conv0_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) { // 서로 다른 img
        //cout << "i = " << i << endl;
        ctxt_block8conv0_out[i] = Conv_parallel(context, pack, eval, 32, 1, 1, 64, 64, ctxt_block7relu1_out[i], block8conv0multiplicands64_64_3_3);
    }

    addBNsummands(context, eval, ctxt_block8conv0_out, block8conv0summands64, 1, 64);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block8conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block8conv0multiplicands64_64_3_3.clear();
    block8conv0multiplicands64_64_3_3.shrink_to_fit();
    block8conv0summands64.clear();
    block8conv0summands64.shrink_to_fit();


    // AppReLU
    cout << "block8relu0 ..." << endl;
    timer.start(" block8relu0 ");
    vector<vector<Ciphertext>> ctxt_block8relu0_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block8conv0_out,ctxt_block8relu0_out);
    /*for (int i = 0; i < 1; ++i) {
        cout << "block8relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 64; ++ch) {
            ApproxReLU(context, eval, ctxt_block8conv0_out[i][ch], ctxt_block8relu0_out[i][ch]);
            eval.levelDown(ctxt_block8relu0_out[i][ch], 5, ctxt_block8relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block8relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "DONE!" << "\n";

    ctxt_block8conv0_out.clear();
    ctxt_block8conv0_out.shrink_to_fit();

    // Second convolution



    vector<double> temp18;
    vector<vector<vector<Plaintext>>> block8conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path18 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block8conv1multiplicands64_64_3_3");
    txtreader(temp18, path18);
    kernel_ptxt(context, temp18, block8conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);
    temp18.clear();
    temp18.shrink_to_fit();

    vector<Plaintext> block8conv1summands64;
    vector<double> temp18a;
    string path18a = "/app/HEAAN-ResNet/kernel/summands/" + string("block8conv1summands64");
    Scaletxtreader(temp18a, path18a, cnst);

    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp18a[i]);
        block8conv1summands64.push_back(ecd.encode(msg, 4, 0));
    }

    temp18a.clear();
    temp18a.shrink_to_fit();



    cout << "block8conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block8relu0_out[0][0].getLevel() << "\n";
    timer.start(" block8conv1 ");
    vector<vector<Ciphertext>> ctxt_block8conv1_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) {
         ctxt_block8conv1_out[i] = Conv_parallel(context, pack, eval, 32, 1, 1, 64, 64, ctxt_block8relu0_out[i], block8conv1multiplicands64_64_3_3);
    }

    addBNsummands(context, eval, ctxt_block8conv1_out, block8conv1summands64, 1, 64);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block8conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block8conv1multiplicands64_64_3_3.clear();
    block8conv1multiplicands64_64_3_3.shrink_to_fit();
    block8conv1summands64.clear();
    block8conv1summands64.shrink_to_fit();




    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "block8add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block8add_out(1, vector<Ciphertext>(64, ctxt_init));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            eval.add(ctxt_block8conv1_out[i][ch], ctxt_block7relu1_out[i][ch], ctxt_block8add_out[i][ch]);
        }
    }
    cout << "DONE!" << "\n";
    ctxt_block7relu1_out.clear();
    ctxt_block7relu1_out.shrink_to_fit();
    ctxt_block8conv1_out.clear();
    ctxt_block8conv1_out.shrink_to_fit();


    cout << "block8relu1 ..." << endl;
    timer.start(" block8relu1 ");
    vector<vector<Ciphertext>> ctxt_block8relu1_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block8add_out,ctxt_block8relu1_out);
    /*for (int i = 0; i < 1; ++i) {
        cout << "block8relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 64; ++ch) {
            ApproxReLU(context, eval, ctxt_block8add_out[i][ch], ctxt_block8relu1_out[i][ch]);
            eval.levelDown(ctxt_block8relu1_out[i][ch], 5, ctxt_block8relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block8relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block8add_out.clear();
    ctxt_block8add_out.shrink_to_fit();
    cout << "RB6 DONE! " << "\n";



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

    vector<Plaintext> block9conv0summands64;
    vector<double> temp19a;
    string path19a = "/app/HEAAN-ResNet/kernel/summands/" + string("block9conv0summands64");
    Scaletxtreader(temp19a, path19a, cnst);

    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp19a[i]);
        block9conv0summands64.push_back(ecd.encode(msg, 4, 0));
    }

    temp19a.clear();
    temp19a.shrink_to_fit();



    cout << "block9conv0 ..." << endl;
    cout << "level of ctxt is " << ctxt_block8relu1_out[0][0].getLevel() << "\n";
    timer.start(" block9conv0 ");
    vector<vector<Ciphertext>> ctxt_block9conv0_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) { // 서로 다른 img
        //cout << "i = " << i << endl;
        ctxt_block9conv0_out[i] = Conv_parallel(context, pack, eval, 32, 1, 1, 64, 64, ctxt_block8relu1_out[i], block9conv0multiplicands64_64_3_3);
    }

    addBNsummands(context, eval, ctxt_block9conv0_out, block9conv0summands64, 1, 64);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block9conv0_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block9conv0multiplicands64_64_3_3.clear();
    block9conv0multiplicands64_64_3_3.shrink_to_fit();
    block9conv0summands64.clear();
    block9conv0summands64.shrink_to_fit();


    // AppReLU
    cout << "block9relu0 ..." << endl;
    timer.start(" block9relu0 ");
    vector<vector<Ciphertext>> ctxt_block9relu0_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block9conv0_out,ctxt_block9relu0_out);
    /*for (int i = 0; i < 1; ++i) {
        cout << "block9relu0 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 64; ++ch) {
            ApproxReLU(context, eval, ctxt_block9conv0_out[i][ch], ctxt_block9relu0_out[i][ch]);
            eval.levelDown(ctxt_block9relu0_out[i][ch], 5, ctxt_block9relu0_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block9relu0_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();
    cout << "DONE!" << "\n";

    ctxt_block9conv0_out.clear();
    ctxt_block9conv0_out.shrink_to_fit();

    // Second convolution

    vector<double> temp20;
    vector<vector<vector<Plaintext>>> block9conv1multiplicands64_64_3_3(64, vector<vector<Plaintext>>(64, vector<Plaintext>(9, ptxt_init)));
    string path20 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block9conv1multiplicands64_64_3_3");
    txtreader(temp20, path20);
    kernel_ptxt(context, temp20, block9conv1multiplicands64_64_3_3, 5, 4, 1, 64, 64, 3, ecd);

    vector<Plaintext> block9conv1summands64;
    vector<double> temp20a;
    string path20a = "/app/HEAAN-ResNet/kernel/summands/" + string("block9conv1summands64");
    Scaletxtreader(temp20a, path20a, cnst);

    for (int i = 0; i < 64; ++i) {
        Message msg(log_slots, temp20a[i]);
        block9conv1summands64.push_back(ecd.encode(msg, 4, 0));
    }

    cout << "block9conv1 ..." << endl;
    cout << "level of ctxt is " << ctxt_block9relu0_out[0][0].getLevel() << "\n";
    timer.start(" block9conv1 ");
    vector<vector<Ciphertext>> ctxt_block9conv1_out(1, vector<Ciphertext>(64, ctxt_init));
    for (int i = 0; i < 1; ++i) {
        ctxt_block9conv1_out[i] = Conv_parallel(context, pack, eval, 32, 1, 1, 64, 64, ctxt_block9relu0_out[i], block9conv1multiplicands64_64_3_3);
    }

    addBNsummands(context, eval, ctxt_block9conv1_out, block9conv1summands64, 1, 64);
    timer.end();
    cout << "DONE!, decrypted message is ... " << "\n";

    dec.decrypt(ctxt_block9conv1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    block9conv1multiplicands64_64_3_3.clear();
    block9conv1multiplicands64_64_3_3.shrink_to_fit();
    block9conv1summands64.clear();
    block9conv1summands64.shrink_to_fit();




    //////////////////////////// Main flow + Residual flow //////////////////////////////////
    cout << "block9add ..." << endl;
    vector<vector<Ciphertext>> ctxt_block9add_out(1, vector<Ciphertext>(64, ctxt_init));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 1; ++i) {
        for (int ch = 0; ch < 64; ++ch) {
            eval.add(ctxt_block9conv1_out[i][ch], ctxt_block8relu1_out[i][ch], ctxt_block9add_out[i][ch]);
        }
    }
    cout << "DONE!" << "\n";
    ctxt_block8relu1_out.clear();
    ctxt_block8relu1_out.shrink_to_fit();
    ctxt_block9conv1_out.clear();
    ctxt_block9conv1_out.shrink_to_fit();


    cout << "block9relu1 ..." << endl;
    timer.start(" block9relu1 ");
    vector<vector<Ciphertext>> ctxt_block9relu1_out(1, vector<Ciphertext>(64, ctxt_init));
    ApproxReLU_bundle80(context,pack,eval,ctxt_block9add_out,ctxt_block9relu1_out);
    /*for (int i = 0; i < 1; ++i) {
        cout << "block9relu1 for (" << i << " , ;)" << "\n";
        for (int ch = 0; ch < 64; ++ch) {
            ApproxReLU(context, eval, ctxt_block9add_out[i][ch], ctxt_block9relu1_out[i][ch]);
            eval.levelDown(ctxt_block9relu1_out[i][ch], 5, ctxt_block9relu1_out[i][ch]);
            if (i == 0 && ch == 0) {
                dec.decrypt(ctxt_block9relu1_out[i][ch], sk, dmsg);
                printMessage(dmsg);
            }
        }
    }*/
    timer.end();

    ctxt_block9add_out.clear();
    ctxt_block9add_out.shrink_to_fit();
    cout << "RB7 DONE! " << "\n";



    // Avg Pool
    cout << "evaluating Avgpool" << "\n";
    vector<Ciphertext> ctxt_avgp_out;
    timer.start(" avgpool * ");
    ctxt_avgp_out = Avgpool(context, pack, eval, ctxt_block9relu1_out[0]);
    timer.end();

    ctxt_block9relu1_out.clear();
    ctxt_block9relu1_out.shrink_to_fit();

    //FC64 setup...
    vector<double> temp21;
    vector<vector<Plaintext>> fclayermultiplicands10_64(10, vector<Plaintext>(64, ptxt_init));
    string path21 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("fclayermultiplicands10_64");
    txtreader(temp21, path21);

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

    double cnst1 = (double)(64.0 / 40.0);
    Scaletxtreader(temp21a, path21a, cnst1);

    for (int i = 0; i < 10; ++i) {
        Message msg(log_slots, temp21a[i]);
        fclayersummands10[i] = ecd.encode(msg, 0, 0);
    }

    temp21a.clear();
    temp21a.shrink_to_fit();


    // FC64
    cout << "evaluating FC64 layer" << endl;

    vector<Ciphertext> ctxt_result;
    timer.start(" FC64 layer * ");
    ctxt_result = FC64(context, pack, eval, ctxt_avgp_out, fclayermultiplicands10_64, fclayersummands10);
    timer.end();

    fclayermultiplicands10_64.clear();
    fclayermultiplicands10_64.shrink_to_fit();
    fclayersummands10.clear();
    fclayersummands10.shrink_to_fit();


    // Last Step; enumerating
    vector<vector<double>> final_result(512, vector<double>(10, 0));

    for (int j = 0; j < 10; ++j) {
        dec.decrypt(ctxt_result[j], sk, dmsg);

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < 32; ++i) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 4; ++l) {
                    final_result[16 * i + 4*k +l][j] = dmsg[1024 * i + 32*k +l].real();
                }
            }
        }

    }

    cout << "Finaly, DONE!!!... output is ..." << "\n";

    

    cout << "[ ";
    for (int i = 0; i < 512; ++i) {
        int max_index = max_element(final_result[i].begin(), final_result[i].end()) - final_result[i].begin();
        cout << max_index << ", ";
    }
    cout << "]\n";

    
    ////////////////////////////////
    ///////////// save file ///////
    //////////////////////////////
    ofstream file("/app/HEAAN-ResNet/output.txt");
    for (int i = 0; i < 512; ++i) {
        int max_index = max_element(final_result[i].begin(), final_result[i].end()) - final_result[i].begin();
        file << max_index << "\n";
    }

    file.close();



    return 0;
}
