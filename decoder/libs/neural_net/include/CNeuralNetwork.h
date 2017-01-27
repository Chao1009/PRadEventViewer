#ifndef C_NEURAL_NETWORK_H
#define C_NEURAL_NETWORK_H

#include <vector>
#include "CNeuronLayer.h"

class CNeuralNetwork
{
public:
	CNeuralNetwork(double factor = 0.2);

    unsigned int CreateNet(unsigned int input,
                           unsigned int output,
                           const std::vector<unsigned int> &hidden_layers);
    unsigned int CreateNet(const char *path);
    void InitializeWeights();
    void SetLearnFactor(double f) {learn_factor = f;};
    void Update(const std::vector<double> &input);
    void Train(const std::vector<double> &input, const std::vector<double> &expect);

    void SaveNet(const char *path) const;
    const std::vector<double> &GetOutput() const {return output;};
    double GetLearnFactor() const {return learn_factor;};

    void BP_Train(const std::vector<double> &in, const std::vector<double> &req);

private:
    std::vector<CNeuronLayer> layers;
    std::vector<double> output;
    double learn_factor;
};

#endif
