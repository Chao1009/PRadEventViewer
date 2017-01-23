#ifndef C_NEURON_LAYER_H
#define C_NEURON_LAYER_H

#include <vector>
#include "CNeuron.h"

class CNeuronLayer
{
public:
    CNeuronLayer(unsigned int con_size, unsigned int size);


    std::vector<double> Output(const std::vector<double> &input) const;
    unsigned int GetInputSize() const {return input_size;};
    std::vector<CNeuron> &GetNeurons() {return neurons;};
    const std::vector<CNeuron> &GetNeurons() const {return neurons;};

private:
    unsigned int input_size;
    std::vector<CNeuron> neurons;
};

#endif
