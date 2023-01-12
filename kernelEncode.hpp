#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "HEaaN/heaan.hpp"
#include "convtools.hpp"
#include "imageEncode.hpp"

namespace {
    using namespace std;
    using namespace HEaaN;
}

// void txtreader(vector<double>& kernel, const string filename) {

//     string line;

//     ifstream input_file(filename);
//     /*if (!input_file.is_open()) {
//         cerr << "Could not open the file - '"
//             << filename << "'" << endl;
//         return EXIT_FAILURE;
//     }*/

//     while (getline(input_file, line)) {
//         double temp = stod(line);
//         kernel.push_back(temp);
//     }

//     /*for (const auto& i : kernel)
//         cout << i << endl;*/

//     input_file.close();
//     return;

// }


void kernel_ptxt(Context context, vector<double>& weight, vector<vector<vector<Plaintext>>>& output, u64 level, u64 gap_in, u64 stride, const int out_ch, const int in_ch, const int ker_size, EnDecoder ecd) {

    if (ker_size == 3) {
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {

                vector<vector<double>> temp(out_ch, vector<double>(in_ch, 0));

                for (int i = 0; i < out_ch; ++i) {
                    for (int j = 0; j < in_ch; ++j) {
                        temp[i][j] = weight[in_ch * 9 * i + 9 * j + 3 * k + l];
                    }
                }

                for (int i = 0; i < out_ch; ++i) {
                    for (int j = 0; j < in_ch; ++j) {
                        weightToPtxt(output[i][j][3 * k + l], level, temp, gap_in, stride, (u64)k, (u64)l, ecd);
                    }
                }
            }
        }
        return;
    }

    if (ker_size == 1) { //output vector is out_ch * in_ch * 1

        vector<vector<double>> temp(out_ch, vector<double>(in_ch, 0));

        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {
                temp[i][j] = weight[in_ch * i + j];
            }
        }

        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {
                weightToPtxt(output[i][j][0], level, temp, gap_in, stride, (u64)1, (u64)1, ecd);
            }
        }
        return;
    }
}

void addBNsummands(Context context, HomEvaluator eval, vector<vector<Ciphertext>>& afterConv, vector<double> summands, const int n, const int ch) {

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ch; ++j) {
            Complex cnst = Complex(summands[j]);
            eval.add(afterConv[i][j], cnst, afterConv[i][j]);
        }
    }

    return;
}
