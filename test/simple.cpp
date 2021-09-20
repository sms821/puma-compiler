/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <iostream>
#include <fstream>
#include <assert.h>
#include <string>
#include <vector>
#include "puma.h"

using namespace std;

int main(int argc, char** argv) {

    Model model = Model::create("simple");
    unsigned int size = 5;
    auto in = InputVector::create(model, "in", size);
    ConstantMatrix matrix = ConstantMatrix::create(model, "constant_", size, size);
    OutputVector out = OutputVector::create(model, "out_", size);

    Vector result = matrix * in;
    out = result;

     // Compile
    model.compile();

    // Bind data
    ModelInstance modelInstance = ModelInstance::create(model);
    float* layer1Weights = new float[size*size];

    //Reading weights from text files
    int i=0;
    std::ifstream wf1;
    wf1.open("simple/wl1.txt");
    while(wf1 >> layer1Weights[i])
    { i++; }
    wf1.close();
    cout << "Read " << i << " weights." << endl;
    //fully_connected_layer_bind(modelInstance, "layer" + std::to_string(1), layer1Weights);
    modelInstance.bind( "constant_", layer1Weights);
    
    modelInstance.generateData();

    // Destroy model
    model.destroy();
    delete[] layer1Weights;

    return 0;

}
