//============================================================================//
// A Neural Network Class                                                     //
//                                                                            //
// Chao Peng                                                                  //
// 01/21/2017                                                                 //
//============================================================================//

#include "CNeuralNetwork.h"



CNeuralNetwork::CNeuralNetwork()
{
    // place holder
}

unsigned int CNeuralNetwork::CreateNet(unsigned int input_size,
                                       unsigned int output_size,
                                       const std::vector<int> &hidden_layers)
{
    // erase original layers
    layers.clear();

    // sanity check
    if(input_size == 0 || output_size == 0)
        return 0;

    // hidden layers
    for(auto &ilayer : hidden_layers)
    {
        if(ilayer <= 0)
            continue;
        layers.emplace_back(input_size, ilayer);
        input_size = ilayer;
    }

    // output layer
    layers.emplace_back(input_size, output_size);

    return layers.size();
}

std::vector<double> CNeuralNetwork::Output(const std::vector<double> &input)
const
{
    // first result
    std::vector<double> result = std::move(layers.front().Output(input));

    for(unsigned int i = 1; i < layers.size(); ++i)
    {
        result = std::move(layers.at(i).Output(result));
    }

    return result;
}

