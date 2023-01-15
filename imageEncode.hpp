#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "HEaaN/heaan.hpp"

namespace {
    using namespace std;
    using namespace HEaaN;
}

void txtreader(vector<double>& kernel, const string filename) {

    string line;
    //ifstream input_file(file_path);
    ifstream input_file(filename);
    /*if (!input_file.is_open()) {
        cerr << "Could not open the file - '"
            << filename << "'" << endl;
        return EXIT_FAILURE;
    }*/

    while (getline(input_file, line)) {
        double temp = stod(line);
        kernel.push_back(temp);
    }

    /*for (const auto& i : kernel)
        cout << i << endl;*/

    input_file.close();
    return;

}

vector<double> slice(const std::vector<double>& input, int a, int b) {
    auto first = input.begin() + a;
    auto last = input.begin() + b+1;
    return std::vector<double>(first, last);
}

//vector<int> merge(vector<int> a, vector<int> b) {
//    vector<int> c;
//    c.reserve(a.size() + b.size()); // preallocate memory
//    c.insert(c.end(), a.begin(), a.end());
//    c.insert(c.end(), b.begin(), b.end());
//}

void imageCompiler(Context context, KeyPack pack, Encryptor enc, u64 level, vector<double>& image, vector<Ciphertext>& output) {

    Message msg1(15), msg2(15), msg3(15);
    auto num_slots = msg1.getSize();
    vector<double> input1, input2, input3;

    Ciphertext ctxt1(context), ctxt2(context), ctxt3(context);
    
    for (int i = 0; i < 32; ++i) {
        vector<double> temp1 = slice(image, 3072 * i, 3072 * i + 1024);
        vector<double> temp2 = slice(image, 3072 * i + 1024, 3072 * i + 2048);
        vector<double> temp3 = slice(image, 3072 * i + 2048, 3072 * i + 3072);


        input1.insert(input1.end(), temp1.begin(), temp1.end());
        input2.insert(input2.end(), temp2.begin(), temp2.end());
        input3.insert(input3.end(), temp3.begin(), temp3.end());

    }
    

    for (size_t i = 0; i < num_slots; ++i) {
        msg1[i].real(input1[i]);
        msg1[i].imag(0.0);

        msg2[i].real(input2[i]);
        msg2[i].imag(0.0);

        msg3[i].real(input3[i]);
        msg3[i].imag(0.0);
    }
    

    Plaintext ptxt1(context), ptxt2(context), ptxt3(context);

    enc.encrypt(msg1, pack, ctxt1, level, 0);
    enc.encrypt(msg2, pack, ctxt2, level, 0);
    enc.encrypt(msg3, pack, ctxt3, level, 0);

    output.push_back(ctxt1);
    output.push_back(ctxt2);
    output.push_back(ctxt3);
    
}
