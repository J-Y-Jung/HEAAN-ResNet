#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "HEaaN/heaan.hpp"
#include "convtools.hpp"

namespace {
    using namespace std;
    using namespace HEaaN;
}

void txtreader(vector<double>& kernel, const string filename) {

    string line;

    ifstream input_file(filename);
    if (!input_file.is_open()) {
        cerr << "Could not open the file - '"
            << filename << "'" << endl;
        return EXIT_FAILURE;
    }

    while (getline(input_file, line)) {
        double temp = stod(line);
        kernel.push_back(temp);
    }

    /*for (const auto& i : kernel)
        cout << i << endl;*/

    input_file.close();
    return EXIT_SUCCESS;

}


void kernel_ptxt(Context context, vector<double>& weight, vector<vector<vector<Plaintext>>>& output, u64 level, u64 gap_in, u64 stride, const int out_ch, const int in_ch, const int ker_size, EnDecoder ecd) {

    if (ker_size == 3) {  //output vector is out_ch * in_ch * 9
        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {

                vector<vector<double>> temp(3, vector<double>(3));

                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++i) {
                        temp[k][l] = weight[27 * i + 9 * j + 3 * k + l];
                    }
                }

                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++i) {
                        Plaintext ptxt(context);

                        weightToPtxt(ptxt, level, temp, gap_in, stride, (u64)k, (u64)l, ecd);

                        output[i][j][k][l] = ptxt;
                    }
                }
            }
        }
    }

    if (ker_size == 1) { //output vector is out_ch * in_ch * 1
        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {

                vector<vector<double>> temp(0, vector<double>(0));
                temp[0][0] = weight[27 * i + 9 * j];

                Plaintext ptxt(context);
                weightToPtxt(ptxt, level, temp, gap_in, stride, (u64)1, (u64)1, ecd);
                output[i][j][0][0] = ptxt;

            }
        }
    }

    return 0;

}

void addBNsummands(Context context, vector<vector<Ciphertext>>& afterConv, vector<double> summands, const int n, const int ch) {
   
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ch; ++j) {
            Complex cnst = Complex(summands[j]);
            eval.add(afterConv[i][j], cnst, afterConv[i][j];
        }
    }

    return 0;
}
