#ifndef C_NEURAL_NETWORK_H
#define C_NEURAL_NETWORK_H

#include <vector>
#include "CNeuronLayer.h"

class CNeuralNetwork
{
public:
	CNeuralNetwork();

    unsigned int CreateNet(unsigned int input_size,
                           unsigned int output_size,
                           const std::vector<unsigned int> &hidden_layers);
    unsigned int CreateNet(const char *path);
    void InitializeWeights();
    void SaveNet(const char *path) const;
    std::vector<double> Output(const std::vector<double> &input) const;

private:
    std::vector<CNeuronLayer> layers;
};

#endif
