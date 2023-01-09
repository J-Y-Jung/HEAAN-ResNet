#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "HEaaN/heaan.hpp"
#include "convtools.hpp"
#include "kernelEncode.hpp"

namespace {
    using namespace std;
    using namespace HEaaN;
}

int main(){

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

    EnDecoder ecd(context);

    //----------------------------------------------------------------------------------------

    vector<double> temp0;
    vector<vector<vector<Plaintext>>> block0conv0multiplicands16_3_3_3;

    string path0 = "./kernel/multiplicands/" + "block0conv0multiplicands16_3_3_3";
    txtreader(temp0, path0);
    kernel_ptxt(context, temp0, block0conv0multiplicands16_3_3_3, 12, 1, 1, 16, 3, 3, ecd);


    vector<double> temp1;
    vector<vector<vector<Plaintext>>> block1conv0multiplicands16_16_3_3;

    string path1 = "./kernel/multiplicands/" + "block1conv0multiplicands16_16_3_3";
    txtreader(temp1, path1);
    kernel_ptxt(context, temp1, block1conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> temp2;
    vector<vector<vector<Plaintext>>> block1conv1multiplicands16_16_3_3;

    string path2 = "./kernel/multiplicands/" + "block1conv1multiplicands16_16_3_3";
    txtreader(temp2, path2);
    kernel_ptxt(context, temp2, block1conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> temp3;
    vector<vector<vector<Plaintext>>> block2conv0multiplicands16_16_3_3;

    string path3 = "./kernel/multiplicands/" + "block2conv0multiplicands16_16_3_3";
    txtreader(temp3, path3);
    kernel_ptxt(context, temp3, block2conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> temp4;
    vector<vector<vector<Plaintext>>> block2conv1multiplicands16_16_3_3;

    string path4 = "./kernel/multiplicands/" + "block2conv1multiplicands16_16_3_3";
    txtreader(temp4, path4);
    kernel_ptxt(context, temp4, block2conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);



    vector<double> temp5;
    vector<vector<vector<Plaintext>>> block3conv0multiplicands16_16_3_3;

    string path5 = "./kernel/multiplicands/" + "block3conv0multiplicands16_16_3_3";
    txtreader(temp5, path5);
    kernel_ptxt(context, temp5, block3conv0multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);



    vector<double> temp6;
    vector<vector<vector<Plaintext>>> block3conv1multiplicands16_16_3_3;

    string path6 = "./kernel/multiplicands/" + "block3conv1multiplicands16_16_3_3";
    txtreader(temp6, path6);
    kernel_ptxt(context, temp6, block3conv1multiplicands16_16_3_3, 12, 1, 1, 16, 16, 3, ecd);


    vector<double> temp7;
    vector<vector<vector<Plaintext>>> block4conv_onebyone_multiplicands32_16_1_1;

    string path7 = "./kernel/multiplicands/" + "block4conv_onebyone_multiplicands32_16_1_1";
    txtreader(temp7, path7);
    kernel_ptxt(context, temp7, block4conv_onebyone_multiplicands32_16_1_1, 12, 1, 2, 32, 16, 1, ecd);


    vector<double> temp8;
    vector<vector<vector<Plaintext>>> block4conv0multiplicands32_16_3_3;

    string path8 = "./kernel/multiplicands/" + "block4conv0multiplicands32_16_3_3";
    txtreader(temp8, path8);
    kernel_ptxt(context, temp8, block4conv0multiplicands32_16_3_3, 12, 1, 2, 32, 16, 3, ecd);


    vector<double> temp9;
    vector<vector<vector<Plaintext>>> block4conv1multiplicands32_32_3_3;

    string path9 = "./kernel/multiplicands/" + "block4conv1multiplicands32_32_3_3";
    txtreader(temp9, path9);
    kernel_ptxt(context, temp9, block4conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);


    vector<double> temp10;
    vector<vector<vector<Plaintext>>> block5conv0multiplicands32_32_3_3;

    string path10 = "./kernel/multiplicands/" + "block5conv0multiplicands32_32_3_3";
    txtreader(temp10, path10);
    kernel_ptxt(context, temp10, block5conv0multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);


    vector<double> temp11;
    vector<vector<vector<Plaintext>>> block5conv1multiplicands32_32_3_3;

    string path1 = "./kernel/multiplicands/" + "block5conv1multiplicands32_32_3_3";
    txtreader(temp11, path11);
    kernel_ptxt(context, temp11, block5conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);


    vector<double> temp12;
    vector<vector<vector<Plaintext>>> block6conv0multiplicands32_32_3_3;

    string path12 = "./kernel/multiplicands/" + "block6conv0multiplicands32_32_3_3";
    txtreader(temp12, path12);
    kernel_ptxt(context, temp12, block6conv0multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);


    vector<double> temp13;
    vector<vector<vector<Plaintext>>> block6conv1multiplicands32_32_3_3;

    string path13 = "./kernel/multiplicands/" + "block6conv1multiplicands32_32_3_3";
    txtreader(temp13, path13);
    kernel_ptxt(context, temp13, block6conv1multiplicands32_32_3_3, 12, 2, 1, 32, 32, 3, ecd);


    vector<double> temp14;
    vector<vector<vector<Plaintext>>> block7conv_onebyone_multiplicands64_32_1_1;

    string path14 = "./kernel/multiplicands/" + "block7conv_onebyone_multiplicands64_32_1_1";
    txtreader(temp14, path14);
    kernel_ptxt(context, temp14, block7conv_onebyone_multiplicands64_32_1_1, 12, 2, 2, 64, 32, 1, ecd);


    vector<double> temp15;
    vector<vector<vector<Plaintext>>> block7conv0multiplicands64_32_3_3;

    string path15 = "./kernel/multiplicands/" + "block7conv0multiplicands64_32_3_3";
    txtreader(temp15, path15);
    kernel_ptxt(context, temp15, block7conv0multiplicands64_32_3_3, 12, 2, 2, 64, 32, 3, ecd);


    vector<double> temp16;
    vector<vector<vector<Plaintext>>> block7conv1multiplicands64_64_3_3;

    string path16 = "./kernel/multiplicands/" + "block7conv1multiplicands64_64_3_3";
    txtreader(temp16, path16);
    kernel_ptxt(context, temp16, block7conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);


    vector<double> temp17;
    vector<vector<vector<Plaintext>>> block8conv0multiplicands64_64_3_3;

    string path17 = "./kernel/multiplicands/" + "block8conv0multiplicands64_64_3_3";
    txtreader(temp17, path17);
    kernel_ptxt(context, temp17, block8conv0multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);


    vector<double> temp18;
    vector<vector<vector<Plaintext>>> block8conv1multiplicands64_64_3_3;

    string path18 = "./kernel/multiplicands/" + "block8conv1multiplicands64_64_3_3";
    txtreader(temp18, path18);
    kernel_ptxt(context, temp18, block8conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);


    vector<double> temp19;
    vector<vector<vector<Plaintext>>> block9conv0multiplicands64_64_3_3;

    string path19 = "./kernel/multiplicands/" + "block9conv0multiplicands64_64_3_3";
    txtreader(temp19, path19);
    kernel_ptxt(context, temp19, block9conv0multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);


    vector<double> temp20;
    vector<vector<vector<Plaintext>>> block9conv1multiplicands64_64_3_3;

    string path20 = "./kernel/multiplicands/" + "block9conv1multiplicands64_64_3_3";
    txtreader(temp20, path20);
    kernel_ptxt(context, temp21, block9conv1multiplicands64_64_3_3, 12, 4, 1, 64, 64, 3, ecd);

}