////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyleft (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "examples.hpp"
#include "Conv.hpp"
#include "oddLazyBSGS.hpp"
#include "MPPacking.hpp"
#include "HEaaNTimer.hpp"
#include "RB+BN.hpp"
#include "RB+BN_parallel.hpp"
#include "kernelEncode.hpp"


namespace {
    using namespace HEaaN;
    using namespace std;
}

int main() {
    HEaaN::HEaaNTimer timer(false);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    if (!HEaaN::isBootstrappableParameter(context)) {
        std::cout << "Bootstrap is not available for parameter "
            << presetNamer(preset) << std::endl;
        return -1;
    }

    std::cout << "Parameter : " << presetNamer(preset) << std::endl
        << std::endl;

    const auto log_slots = getLogFullSlots(context);

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... " << std::endl;
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl << std::endl;

    /*
    You should perform makeBootstrappble function
    before generating evaluation keys and constucting HomEvaluator class.
    */
    HEaaN::makeBootstrappable(context);

    std::cout << "Generate commonly used keys (mult key, rotation keys, "
        "conjugation key) ... "
        << std::endl;
    keygen.genCommonKeys();
    std::cout << "done" << std::endl << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);

    /*
    HomEvaluator constructor pre-compute the constants for bootstrapping.
    */
    std::cout << "Generate HomEvaluator (including pre-computing constants for "
        "bootstrapping) ..."
        << std::endl;
    timer.start("* ");
    HEaaN::HomEvaluator eval(context, pack);
    timer.end();

    EnDecoder ecd(context);
    // ///////////// Preset ///////////////////
    // std::int rb_num;
    // rb_num = 16;
    
    cout << "Image Loading ..." << "\n";
    vector<vector<Ciphertext>> imageVec;


    string str = "/app/HEAAN-ResNet/image/image_" + to_string(1) + string(".txt");
    vector<double> temp;
    txtreader(temp, str);
    vector<Ciphertext> out;
    imageCompiler(context, pack, enc, 5, temp, out);
    imageVec.push_back(out);


    cout << "DONE, test for image encode ..." << "\n";

    Message dmsg;
    dec.decrypt(imageVec[0][0], sk, dmsg);
    printMessage(dmsg);

    cout << "DONE" << "\n";


    Plaintext ptxt_init(context);


    // RB 1 - 1
    vector<double> temp1;
    vector<vector<vector<Plaintext>>> block1conv0multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path1 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block1conv0multiplicands16_16_3_3");
    txtreader(temp1, path1);
    kernel_ptxt(context, temp1, block1conv0multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);

    vector<Plaintext> block1conv0summands16;
    vector<double> temp1a;
    string path1a = "/app/HEAAN-ResNet/kernel/summands/" + string("block1conv0summands16");
    txtreader(temp1a, path1a);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp1a[i]);
        block1conv0summands16.push_back(ecd.encode(msg, 4, 0));
    }




    // RB 1 - 2
    vector<double> temp2;
    vector<vector<vector<Plaintext>>> block1conv1multiplicands16_16_3_3(16, vector<vector<Plaintext>>(16, vector<Plaintext>(9, ptxt_init)));
    string path2 = "/app/HEAAN-ResNet/kernel/multiplicands/" + string("block1conv1multiplicands16_16_3_3");
    txtreader(temp2, path2);
    kernel_ptxt(context, temp2, block1conv1multiplicands16_16_3_3, 5, 1, 1, 16, 16, 3, ecd);


    vector<Plaintext> block1conv1summands16;
    vector<double> temp2a;
    string path2a = "/app/HEAAN-ResNet/kernel/summands/" + string("block1conv1summands16");
    txtreader(temp2a, path2a);

    for (int i = 0; i < 16; ++i) {
        Message msg(log_slots, temp2a[i]);
        block1conv1summands16.push_back(ecd.encode(msg, 4, 0));
    }


    // RB 1
    cout << "RB 1 ..." << endl;
    timer.start(" RB 1 ");
    vector<vector<Ciphertext>> ctxt_RB1_out;
    ctxt_RB1_out = RB(context, pack, eval, 0, imageVec, block1conv0multiplicands16_16_3_3, block1conv1multiplicands16_16_3_3,
        block1conv0summands16, block1conv1summands16);
    timer.end();
    cout << "DONE, test for image encode ..." << "\n";

    dec.decrypt(ctxt_RB1_out[0][0], sk, dmsg);
    printMessage(dmsg);


    // parallel_RB 1
    cout << "RB 1 ..." << endl;
    timer.start(" RB 1 ");
    vector<vector<Ciphertext>> ctxt_RB1_out1;
    ctxt_RB1_out1 = RB_parallel(context, pack, eval, 0, imageVec, block1conv0multiplicands16_16_3_3, block1conv1multiplicands16_16_3_3,
        block1conv0summands16, block1conv1summands16);
    timer.end();
    cout << "DONE, test for image encode ..." << "\n";

    dec.decrypt(ctxt_RB1_out1[0][0], sk, dmsg);
    printMessage(dmsg);


    //memeory delete    
    block1conv0multiplicands16_16_3_3.clear();
    block1conv0multiplicands16_16_3_3.shrink_to_fit();
    block1conv1multiplicands16_16_3_3.clear();
    block1conv1multiplicands16_16_3_3.shrink_to_fit();
    block1conv0summands16.clear();
    block1conv0summands16.shrink_to_fit();
    block1conv1summands16.clear();
    block1conv1summands16.shrink_to_fit();

    return 0;
}
