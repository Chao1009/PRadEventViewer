#ifndef C_NEURON_H
#define C_NEURON_H

#include <vector>

class CNeuron
{
public:
    CNeuron(unsigned int size);

    void SetWeights(const std::vector<double> &w);
    double Output(const std::vector<double> &input) const;
    std::vector<double> &GetWeights() {return weights;};
    const std::vector<double> &GetWeights() const {return weights;};

private:
    double sigmoid(const double &a, const double &p) const;

private:
    std::vector<double>	weights;
};

#endif
