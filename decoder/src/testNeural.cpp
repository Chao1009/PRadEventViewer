//============================================================================//
// An example showing how to use the neural network                           //
//                                                                            //
// Chao Peng                                                                  //
// 01/24/2017                                                                 //
//============================================================================//

#include "CNeuralNetwork.h"
#include <iostream>

using namespace std;

int main(int /*argc*/, char * /*argv*/ [])
{
    CNeuralNetwork my_net;
//    my_net.CreateNet(5, 1, {3});
//    my_net.InitializeWeights();
    my_net.CreateNet("save_net.dat");
    auto result = my_net.Output({0.1, 0.2, 0.3, 0.4, 0.5});
//    my_net.SaveNet("save_net.dat");
    cout << result.at(0) << endl;
    return 0;
}
