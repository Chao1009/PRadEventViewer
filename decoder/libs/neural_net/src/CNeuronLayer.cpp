//============================================================================//
// A layer of neurons                                                         //
//                                                                            //
// Chao Peng                                                                  //
// 01/21/2017                                                                 //
//============================================================================//
#include "CNeuronLayer.h"
#include <iostream>


CNeuronLayer::CNeuronLayer(unsigned int con_size, unsigned int size)
: input_size(con_size)
{
	for(unsigned int i = 0; i < size; ++i)
    {
		neurons.emplace_back(con_size);
    }
}

std::vector<double> CNeuronLayer::Output(const std::vector<double> &input)
const
{
    std::vector<double> result;
    if(input.size() != input_size)
    {
        std::cerr << "Unmatched input size, expecting " << input_size << " inputs."
                  << std::endl;
        result.resize(neurons.size(), 0.);
        return result;
    }

    result.reserve(neurons.size());
    for(auto &neuron : neurons)
    {
        result.push_back(neuron.Output(input));
    }

    return result;
}

