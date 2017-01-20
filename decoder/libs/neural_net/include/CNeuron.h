#ifndef C_NEURON_H
#define C_NEURON_H

#include <vector>

class CNeuron
{
public:
    CNeuron(unsigned int size);

    double Output(const std::vector<double> &input) const;

private:
    double sigmoid(const double &a, const double &p) const;

private:
    std::vector<double>	weights;
};

#endif
