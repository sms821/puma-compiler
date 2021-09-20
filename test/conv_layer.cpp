/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "puma.h"
#include "conv_layer.h"

int main(int argc, char** argv) {

    Model model = Model::create("conv_layer");

    // Process parameters
    unsigned int in_size_x = 14;
    unsigned int in_size_y = 14;
    unsigned int in_channels = 32;
    unsigned int out_channels = 64;
    unsigned int k_size_x = 3;
    unsigned int k_size_y = 3;
    if(argc == 7) {
        in_size_x = atoi(argv[1]);
        in_size_y = atoi(argv[2]);
        in_channels = atoi(argv[3]);
        out_channels = atoi(argv[4]);
        k_size_x = atoi(argv[5]);
        k_size_y = atoi(argv[6]);
    }

    // Input stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Output stream
    unsigned int out_size_x = in_size_x;
    unsigned int out_size_y = in_size_y;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    // Layer
    out_stream = conv_layer(model, "", k_size_x, k_size_y, in_size_x, in_size_y, in_channels, out_channels, in_stream);

    // Compile
    model.compile();

    // Bind data
    ModelInstance modelInstance = ModelInstance::create(model);
    float* layer1Weights = new float[k_size_x * k_size_y * in_channels * out_channels];
    
    //Reading weights from text files
    int i=0;
    std::ifstream wf1;
    wf1.open("conv_layer_weights/wl1.txt");
    while(wf1 >> layer1Weights[i])
    { i++; }
    wf1.close();
    std::cout << "Read " << i << " weights." << std::endl;

    conv_layer_bind(modelInstance, "", layer1Weights);
    
    //modelInstance.bind("layer" + std::to_string(1) + "mat", layer1Weights);
    
    modelInstance.generateData();

    // Destroy model
    model.destroy();
    delete[] layer1Weights;

    return 0;

}

