//============================================================================//
// Basic neuron unit in a neural network                                      //
//                                                                            //
// Chao Peng                                                                  //
// 01/21/2017                                                                 //
//============================================================================//

#include "CNeuron.h"
#include <iostream>
#include <cmath>



// constructor
CNeuron::CNeuron(unsigned int size)
{
    // connections + bias
    weights.resize(size + 1, 0.);
}

// set the weights for this neuron
void CNeuron::SetWeights(const std::vector<double> &w)
{
    if(w.size() != weights.size())
    {
        std::cout << "Unmatched input size " << w.size()
                  << " and weights size " << weights.size()
                  << ", the setting may be incorrect."
                  << std::endl;
    }

    for(unsigned int i = 0; i < w.size() && i < weights.size(); ++i)
    {
        weights.at(i) = w.at(i);
    }
}

// give the neuron an input array and get its output
double CNeuron::Output(const std::vector<double> &input)
const
{
    if(input.size() != weights.size() - 1)
    {
        std::cerr << "Unmatched input dimension and connections" << std::endl;
        return 0.;
    }

    double result = 0.;
    for(unsigned int i = 0; i < input.size(); ++i)
    {
        result += weights.at(i)*input.at(i);
    }

    // bias part
    result -= weights.back();

    return sigmoid(result, 1.0);
}

// sigmoid function for output
inline double CNeuron::sigmoid(const double &a, const double &p)
const
{
	return 1./(1. + std::exp(-a/p));
}

