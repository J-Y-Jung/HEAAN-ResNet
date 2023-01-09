#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "HEaaN/heaan.hpp"
#include "imageEncode.hpp"

namespace {
    using namespace std;
    using namespace HEaaN;
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
    HomEvaluator eval(context, pack);

    vector<vector<Ciphertext>> imageVec;

    for (int i = 1; i < 313; ++i) {
        string str = "./cifar10/image_" + to_string(i) + ".txt";
        vector<double> temp;

        txtreader(temp, str);

        vector<Ciphertext> out;
        imageCompiler(context, pack, enc, temp, out);
        imageVec.push_back(out);

    }

    string str = "./cifar10/image_" + to_string(313) + ".txt";
    vector<double> temp;
    txtreader(temp, str);
    for (int i = 0; i < 49152; ++i) temp.push_back(0);

    vector<Ciphertext> out;
    imageCompiler(context, pack, enc, temp, out);
    imageVec.push_back(out);


    return 0;
}