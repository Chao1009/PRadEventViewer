//============================================================================//
// Basic neuron unit in a neural network                                      //
//                                                                            //
// Chao Peng                                                                  //
// 01/21/2017                                                                 //
//============================================================================//

#include "CNeuron.h"
#include <iostream>
#include <cmath>


CNeuron::CNeuron(unsigned int size)
{
    weights.resize(size + 1, 0.);
}

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

// sigmoid function
inline double CNeuron::sigmoid(const double &a, const double &p)
const
{
	return 1./(1. + std::exp(-a/p));
}

