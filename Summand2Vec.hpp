
#include <iostream>
#include <fstream>
#include <vector>
#include <istream>
#include <sstream>

using namespace std;

void Summands2Vec(string path , int num_channel , vector<double> &summand){
    
    //file read -> vector 
    string line;
    fstream fs;
    vector<string> lines;

    fs.open(path,ios::in); //input file_Path
    
    //lines save each row element in file
    while(getline(fs, line)){
        lines.push_back(line);
    }

    for(int i = 0 ; i < num_channel; i++){   
        summand[i] = stod(lines[i]);
    }

    fs.close();
}