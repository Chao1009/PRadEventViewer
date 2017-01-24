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
    connections.resize(size);
}

CNeuron::CNeuron(std::vector<CNeuron> &neurons)
{
    for(auto &neuron : neurons)
    {
        connections.emplace_back(0., &neuron);
    }
}

void CNeuron::Connect(unsigned int idx, CNeuron *n)
{
    if(idx < connections.size())
    {
        connections.at(idx).neuron = n;
    }
}

void CNeuron::Connect(std::vector<CNeuron> &neurons)
{
    for(unsigned int i = 0; i < neurons.size(); ++i)
    {
        Connect(i, &neurons.at(i));
    }
}

// set the weights for this neuron
void CNeuron::SetWeights(const std::vector<double> &w)
{
    if(w.size() != connections.size() + 1)
    {
        std::cerr << "Unmatched input size " << w.size()
                  << " and weights size " << connections.size() + 1
                  << ", abort weight setting."
                  << std::endl;
        return;
    }

    for(unsigned int i = 0; i < connections.size(); ++i)
    {
        connections.at(i).weight = w.at(i);
    }

    bias = w.back();
}

// give the neuron an input array and update its output
void CNeuron::Update(const std::vector<double> &input)
{
    signal = 0.;
    for(unsigned int i = 0; i < input.size(); ++i)
    {
        signal += connections.at(i).weight*input.at(i);
    }

    // bias part
    signal -= bias;

    signal = sigmoid(signal, 1.0);
}

// let the neuron update its signal from connected neurons
void CNeuron::Update()
{
    signal = 0.;
    for(auto &conn : connections)
    {
        if(conn.neuron)
        {
            signal += conn.neuron->signal*conn.weight;
        }
    }

    signal = sigmoid(signal, 1.0);
}

// get weights from all connections and bias
std::vector<double> CNeuron::GetWeights()
const
{
    std::vector<double> res;
    res.reserve(connections.size() + 1);
    for(auto conn : connections)
    {
        res.push_back(conn.weight);
    }
    res.push_back(bias);

    return res;
}

// sigmoid function for output
inline double CNeuron::sigmoid(const double &a, const double &p)
const
{
	return 1./(1. + std::exp(-a/p));
}

inline double CNeuron::dsigmoid(const double &a, const double &p)
const
{
    double expap = std::exp(-a/p);
    return 1./p * expap/(1. + expap)/(1. + expap);
}
