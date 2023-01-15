#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "HEaaN/heaan.hpp"
#include "convtools.hpp"
#include "imageEncode.hpp"

 void Scaletxtreader(vector<double>& kernel, const string filename, const double cnst) {

     string line;
     ifstream input_file(filename);

     while (getline(input_file, line)) {
         double temp = stod(line);
         double temp1 = cnst* temp;
         kernel.push_back(temp1);
     }

     input_file.close();
     return;

 }


void kernel_ptxt(Context context, vector<double>& weight, vector<vector<vector<Plaintext>>>& output, u64 level, u64 gap_in, u64 stride, const int out_ch, const int in_ch, const int ker_size, EnDecoder ecd) {


    if (ker_size == 3) {
        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        weightToPtxt(output[i][j][3 * k + l], level, weight[in_ch * 9 * i + 9 * j + 3 * k + l], gap_in, k , l, ecd);
                    }
                }
            }
        }
        return;
    }
    
    if (ker_size == 1) { //output vector is out_ch * in_ch * 1
        for (int i = 0; i < out_ch; ++i) {
            for (int j = 0; j < in_ch; ++j) {
                weightToPtxt(output[i][j][0], level, weight[in_ch * i + j], gap_in, 1, 1, ecd);
            }
        }
        return;
    }

}


void addBNsummands(Context context, HomEvaluator eval, vector<vector<Ciphertext>>& afterConv, vector<Plaintext>& summands, const int n, const int ch) {

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ch; ++j) {
            eval.add(afterConv[i][j], summands[j], afterConv[i][j]);
        }
    }

    return;
}
