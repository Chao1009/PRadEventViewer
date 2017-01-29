//============================================================================//
// An example showing how to train the neural network for cosmic rejection.   //
//                                                                            //
// Chao Peng                                                                  //
// 01/27/2017                                                                 //
//============================================================================//

#include "CNeuralNetwork.h"
#include "cosmicEval.h"
#include "PRadHyCalSystem.h"
#include "PRadDataHandler.h"
#include "PRadDSTParser.h"
#include "PRadBenchMark.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

#define PROGRESS_COUNT 1000

using namespace std;

void NeuralTrain(CNeuralNetwork &net, PRadHyCalSystem &sys, string path, bool cosmic);
vector<PRadHyCalModule*> modules;

void Helper()
{
    cout << "usage: " << endl
         << setw(10) << " " << "neuralTrain <cosmic_data> <good_data>"
         << endl
         << "options:" << endl
         << setw(10) << "-n " << "<file_path>: "
         << "create network from file, a new network will be created by default."
         << endl
         << setw(10) << "-s " << "<file_path>: "
         << "set the path to save the trained network, save to \"saved.net\" by default "
         << endl
         << setw(10) << "-f " << "<value>: "
         << "define learning factor, 0.1 is the default value."
         << endl
         << setw(10) << "-i " << "<value>: "
         << "set the iteration of training process, 100 is the default value."
         << endl
         << setw(10) << "-h " << ": "
         << "see this helper information."
         << endl;
}

int main(int argc, char *argv[])
{
    string net_path, save_path, argstr[2];
    double learn_factor = 0.1;
    int learn_iter = 100, run_arg = 0;

    save_path = "saved.net";

    for(int i = 1; i < argc; ++i)
    {
        char* ptr = argv[i];
        if(*(ptr++) == '-')
        {
            switch(*(ptr++))
            {
            case 'n':
                net_path = argv[++i];
                break;
            case 's':
                save_path = argv[++i];
                break;
            case 'f':
                learn_factor = strtod(argv[++i], nullptr);
                break;
            case 'i':
                learn_iter = atoi(argv[++i]);
                break;
            case 'h':
            default:
                Helper();
                return -1;
            }
        }
        else
        {
            if(run_arg < 2)
                argstr[run_arg] = argv[i];
            run_arg++;
        }
    }

    if(run_arg != 2) {
        Helper();
        return -1;
    }

    string cosmic_file = argstr[0];
    string data_file = argstr[1];
    CNeuralNetwork my_net(learn_factor);

    if(net_path.empty())
    {
        // create net with dimensions
        // 2 hidden layers have 5 and 3 neurons respectively
        std::vector<unsigned int> hidden = {20, 10, 5, 3};
        // 5 inputs and 3 outputs with hidden layers
        my_net.CreateNet(6, 1, hidden);
        // initialize the weights with random values
        my_net.InitializeWeights();
    }
    else {
        // or create net from saved network data
        // exit if fail to create
        if(my_net.CreateNet(net_path.c_str()) == 0) {
            cout << "Failed to create the network." << endl;
            return -1;
        }
    }

    PRadHyCalSystem sys;
    sys.Configure("config/hycal.conf");

    for(int i = 1; i <= learn_iter; ++i)
    {
        cout << "Training iteration " << i << "." << endl;
        NeuralTrain(my_net, sys, cosmic_file, true);
        NeuralTrain(my_net, sys, data_file, false);
    }

    my_net.SaveNet(save_path.c_str());
    return 0;
}

void NeuralTrain(CNeuralNetwork &net, PRadHyCalSystem &sys, string path, bool cosmic)
{
    vector<double> expect;

    cout << "Start to train the network with file "
         << "\"" << path << "\" ";
    if(cosmic)
    {
        cout << "(cosmic)." << endl;
        expect.push_back(0.8);
    }
    else
    {
        cout << "(good events)." << endl;
        expect.push_back(0.2);
    }

    PRadDSTParser dst;
    dst.OpenInput(path);

    int count = 0;
    double average = 0.;
    PRadBenchMark timer;
    while(dst.Read())
    {
        if(dst.EventType() == PRadDSTParser::Type::event)
        {

            auto &event = dst.GetEvent();

            if(!event.is_physics_event())
                continue;

            if((++count)%PROGRESS_COUNT == 0) {
                cout << "----------event " << count
                     << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
                     << "\r" << flush;
            }

            auto param = AnalyzeEvent(&sys, event, 3.0);
            if(param.group_size == 0)
                continue;

            net.BP_Train(param.GetParamList(), expect);

            average += net.GetOutput().at(0);
        }
    }

    cout << "----------event " << count
         << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
         << endl;
    cout << "Finished training from file " << path << endl;
    cout << "Average output is " << average/(double)count << endl;
}
