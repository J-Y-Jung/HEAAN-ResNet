
#include <iostream>
#include "Weight2Vec.hpp"

using namespace std;

int main(){

    int num_output_channel = 64;
    int num_input_channel = 32;
    int kernel_size = 1;

    vector<vector<vector<double>>> kernel_weight(num_output_channel, vector<vector<double>>(num_input_channel, vector<double>(kernel_size * kernel_size,0)));

    //Path means file path which is located in program running computer.
    Weight2Vec(Path,num_output_channel,num_input_channel,kernel_size,kernel_weight);

    //print vector elements.
     for(int i = 0 ; i < num_output_channel; i++){
        for(int j = 0 ; j < num_input_channel ; j++){
            for(int k = 0 ; k < kernel_size*kernel_size ; k++){
                    cout << "i : " << i+1 << " j : "<< j+1 << " k : " << k+1 << " " ;
                    cout <<  kernel_weight[i][j][k] <<endl;
            }
            cout << std::endl;
        }
    }
    return 0;



}
