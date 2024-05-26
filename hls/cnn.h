#ifndef CNN_H_
#define CNN_H_

typedef float feature_map_t;
typedef float weight_t;

#define STRIDE      2
#define PADDING     0
#define KERNEL_SIZE 5


/**------------------------------------------------------------------------------------------------
 *                                         BUFFERS
 *  Output buffer    (Y_buff) [batch size][output channels][signal length after conv]
 *  Input buffer     (X_buff) [batch size][input channels][input signal length]
 *  Weight buffer    (W_buff) [output channels][input channels][kernel size]
 *  Bias buffer      (B_buff) [output channels]
 * 
 *  Max Pooling Output Buffer (Y_maxpool_buff) [batch size][output channels][reduced signal length after pooling]
 *  Flattening Output Buffer  (out) [batch size][total features]
 * 
 *  Dense Layer Output Buffers:
 *    - Dense1 Output (fixp_dense1_output) [batch size][neurons in dense1]
 *    - Dense2 Output (fixp_dense2_output) [batch size][neurons in dense2]
 * 
 *  Dense Layer Weight Buffers:
 *    - Dense1 Weights (fixp_dense1_weights) [neurons in dense1][input features to dense1]
 *    - Dense2 Weights (fixp_dense2_weights) [neurons in dense2][input features to dense2]
 * 
 *  Dense Layer Bias Buffers:
 *    - Dense1 Bias (fixp_dense1_bias) [neurons in dense1]
 *    - Dense2 Bias (fixp_dense2_bias) [neurons in dense2]
 * 
 *  CNN Buffers:
 *    - Input Feature Map (input_feature_map) [batch size][input channels][input signal length]
 *    - Convolution Weights and Biases:
 *      - Layer 1 Weights (conv_layer_weights1) [output channels][input channels][kernel size]
 *      - Layer 1 Bias (conv_layer_bias1) [output channels]
 *      - Layer 2 Weights (conv_layer_weights2) [output channels][input channels][kernel size]
 *      - Layer 2 Bias (conv_layer_bias2) [output channels]
 *      - Layer 3 Weights (conv_layer_weights3) [output channels][input channels][kernel size]
 *      - Layer 3 Bias (conv_layer_bias3) [output channels]
 *      - Layer 4 Weights (conv_layer_weights4) [output channels][input channels][kernel size]
 *      - Layer 4 Bias (conv_layer_bias4) [output channels]
 * 
 *    - Final Convolution Output (conv_layer_output_feature) [batch size][final output features]
 *------------------------------------------------------------------------------------------------**/

/**------------------------------------------------------------------------------------------------
 *                                         CONVOLUTIONAL LAYERS
 *------------------------------------------------------------------------------------------------**/
void conv1d_1(
    feature_map_t   Y_buff[1][32][187],
    feature_map_t   X_buff[1][1][187],
    weight_t        W_buff[32][1][5],
    weight_t        B_buff[32]
);

void conv1d_2(
    feature_map_t   Y_buff[1][32][92],
    feature_map_t   X_buff[1][32][92],
    weight_t        W_buff[32][32][5],
    weight_t        B_buff[32]
    
);
void conv1d_3(
    feature_map_t   Y_buff[1][32][44],
    feature_map_t   X_buff[1][32][44],
    weight_t        W_buff[32][32][5],
    weight_t        B_buff[32]
    
);
void conv1d_4(
    feature_map_t   Y_buff[1][32][20],
    feature_map_t   X_buff[1][32][20],
    weight_t        W_buff[32][32][5],
    weight_t        B_buff[32]
    
);

/**------------------------------------------------------------------------------------------------
 *                                         MAXPOOLING LAYERS
 *------------------------------------------------------------------------------------------------**/
 void max_pooling1(
    feature_map_t Y_buff[1][32][187],
    feature_map_t Y_maxpool_buff[1][32][92]
     
 );

void max_pooling2(
    feature_map_t Y_buff[1][32][92],
    feature_map_t Y_maxpool_buff[1][32][44]
     
 );
 void max_pooling3(
    feature_map_t Y_buff[1][32][44],
    feature_map_t Y_maxpool_buff[1][32][20]
     
 );
 void max_pooling4(
    feature_map_t Y_buff[1][32][20],
    feature_map_t Y_maxpool_buff[1][32][8]
     
 );
 void max_pooling5(
    feature_map_t Y_buff[1][32][8],
    feature_map_t Y_maxpool_buff[1][32][2]
     
 );

 /**------------------------------------------------------------------------------------------------
 *                                         FLATTEN LAYER
 *------------------------------------------------------------------------------------------------**/
void flatten(
    feature_map_t in[1][32][2],
    feature_map_t out[1][64]
);


 /**------------------------------------------------------------------------------------------------
 *                                         DENSE LAYERS
 *------------------------------------------------------------------------------------------------**/

void dense1(
    feature_map_t conv_layer_output_feature_flat [1][64],
    feature_map_t dense_1_bias[32],
    feature_map_t dense_1_weights[32][64],
    feature_map_t dense_1_ouput[1][32]
);


void dense2(
    feature_map_t dense_1_ouput[1][32],
    feature_map_t dense_2_bias[5],
    feature_map_t dense_2_weights[5][32],
    feature_map_t dense_2_ouput[1][5]
);



 /**------------------------------------------------------------------------------------------------
 *                                         CNN
 *------------------------------------------------------------------------------------------------**/
void cnn(
    feature_map_t   input_feature_map[1][1][187],
    weight_t        conv_layer_weights1[32][1][5],
    weight_t        conv_layer_bias1[32],
    weight_t        conv_layer_weights2[32][32][5],
    weight_t        conv_layer_bias2[32],
    weight_t        conv_layer_weights3[32][32][5],
    weight_t        conv_layer_bias3[32],
    weight_t        conv_layer_weights4[32][32][5],
    weight_t        conv_layer_bias4[32],
    weight_t        dense1_weights[32][64],
    weight_t        dense1_bias[32],
    weight_t        dense2_weights[5][32],
    weight_t        dense2_bias[5],
    weight_t        conv_layer_output_feature[5]
    
);



#endif
