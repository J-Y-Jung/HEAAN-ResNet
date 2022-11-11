
#include <iostream>
#include <fstream>
#include <vector>
#include <istream>
#include <sstream>

using namespace std;

void Weight2Vec(string path , int num_output_channel , int num_input_channel, int kernel_size,
        vector<vector<vector<double>>> &kernel_weight){
    
    //file read -> vector 
    string line;
    fstream fs;
    vector<string> lines;

    fs.open(path,ios::in); //input file_Path
    
    //lines save each row element in file
    while(getline(fs, line)){
        lines.push_back(line);
    }

   
    int total_num = num_output_channel * num_input_channel * kernel_size * kernel_size;
    for(int i = 0 ; i < num_output_channel; i++){
        for(int j = 0 ; j < num_input_channel ; j++){
            for(int k = 0 ; k < kernel_size*kernel_size ; k++){
                    int a = kernel_size * kernel_size * (num_input_channel * i +  j); 
                    kernel_weight[i][j][k] = std::stod(lines[a+k]);
            }
        }
    }


    fs.close();
}
