#include "cnn.h"
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

//feature map -> output from the filter

feature_map_t conv_layer_output_feature_map1    [1][32][187];
feature_map_t conv_layer_output_feature_map1_max[1][32][92];
feature_map_t conv_layer_output_feature_map2    [1][32][92];
feature_map_t conv_layer_output_feature_map2_max[1][32][44];
feature_map_t conv_layer_output_feature_map3    [1][32][44];
feature_map_t conv_layer_output_feature_map3_max[1][32][20];
feature_map_t conv_layer_output_feature_map4    [1][32][20];
feature_map_t conv_layer_output_feature_map4_max[1][32][8];
feature_map_t conv_layer_output_feature_map5_max[1][32][2];
feature_map_t dense1_ouput[1][32];
feature_map_t conv_layer_output_feature_flat[1][64];

void conv1d_1(feature_map_t Y_buff[1][32][187], feature_map_t X_buff[1][1][187], weight_t W_buff[32][1][5], weight_t B_buff[32]){
#pragma HLS pipeline off
    conv1d_1a:
    for (int i=0; i<32; i++) 
    {
        conv1d_1b:
        for (int j=0; j<187; j++)
        {

            conv1d_1c:
            for (int k=0; k<5; k++) {
                
                if (j+k <2 || i+k > 188) {

                    Y_buff[0][i][j] += 0;

                }
                else {
                    Y_buff[0][i][j] += X_buff[0][0][j+k-2]*W_buff[i][0][k];
                    
                }

            }

            //add ReLU
            Y_buff[0][i][j] += B_buff[i]; //adds bias

            if (Y_buff[0][i][j] < 0)
            {
                Y_buff[0][i][j] = 0; // Set to zero negative values
            }
            else
            {
                Y_buff[0][i][j] = Y_buff[0][i][j]; 
            }



    }}



}