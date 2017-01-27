//============================================================================//
// An example showing how to use the neural network                           //
//                                                                            //
// Chao Peng                                                                  //
// 01/24/2017                                                                 //
//============================================================================//

#include "CNeuralNetwork.h"
#include <iostream>

#define NEW_NET

using namespace std;

int main(int /*argc*/, char * /*argv*/ [])
{
    CNeuralNetwork my_net;

#ifdef NEW_NET
    // create net with dimensions
    // 2 hidden layers have 3 and 2 neurons respectively
    std::vector<unsigned int> hidden = {3, 2};
    // 5 inputs and 3 outputs with hidden layers
    my_net.CreateNet(5, 3, hidden);
    // initialize the weights with random values
    my_net.InitializeWeights();
#else
    // or create net from saved network data
    my_net.CreateNet("save_net.dat");
#endif

    // set input and expected output
    std::vector<double> input = {1, 2, 3, 4, 5};
    std::vector<double> expect = {0.3, 0.6, 0.9};

    my_net.Update(input);

    // output with random weights
    cout << my_net.GetOutput().at(0) << ", "
         << my_net.GetOutput().at(1) << ", "
         << my_net.GetOutput().at(2)
         << endl;

    // train it 2000 times
    int count = 2000;
    while(count-- > 0)
    {
        my_net.BP_Train(input, expect);
    }

    // output after training
    cout << my_net.GetOutput().at(0) << ", "
         << my_net.GetOutput().at(1) << ", "
         << my_net.GetOutput().at(2)
         << endl;

    // save the result
    my_net.SaveNet("save_net.dat");
    return 0;
}
