//============================================================================//
// An example showing how to train the neural network for cosmic rejection.   //
//                                                                            //
// Chao Peng                                                                  //
// 01/27/2017                                                                 //
//============================================================================//

#include "CNeuralNetwork.h"
#include "PRadHyCalSystem.h"
#include "PRadDataHandler.h"
#include "PRadDSTParser.h"
#include "PRadBenchMark.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

#define PROGRESS_COUNT 1000

using namespace std;

void NeuralTrain(CNeuralNetwork &net, PRadHyCalSystem &sys, string path, string path2);
void FillInput(PRadHyCalSystem &sys, vector<double> &input, const EventData &event);
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
         << "define learning factor, 0.01 is the default value."
         << endl
         << setw(10) << "-i " << "<value>: "
         << "set the iteration of training process, 2 is the default value."
         << endl
         << setw(10) << "-h " << ": "
         << "see this helper information."
         << endl;
}

int main(int argc, char *argv[])
{
    string net_path, save_path, argstr[2];
    double learn_factor = 0.01;
    int learn_iter = 2, run_arg = 0;

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
        std::vector<unsigned int> hidden = {500, 35, 5};
        // 5 inputs and 3 outputs with hidden layers
        my_net.CreateNet(1728, 1, hidden);
        // initialize the weights with random values
        my_net.InitializeWeights();
    }
    else {
        // or create net from saved network data
        // exit if fail to create
        if(my_net.CreateNet(net_path.c_str()) == 0)
            return -1;
    }

    PRadHyCalSystem sys;
    sys.Configure("config/hycal.conf");

    // organize the modules geometrically
    modules = sys.GetDetector()->GetModuleList();

    sort(modules.begin(), modules.end(),
         [] (PRadHyCalModule *m1, PRadHyCalModule *m2)
         {
            if(m1->GetY() == m2->GetY())
                return m1->GetX() < m2->GetX();
            else
                return m1->GetY() < m2->GetY();
         });

    while(learn_iter--)
    {
        NeuralTrain(my_net, sys, cosmic_file, data_file);
    }

    my_net.SaveNet(save_path.c_str());
    return 0;
}

void NeuralTrain(CNeuralNetwork &net, PRadHyCalSystem &sys, string path, string path2)
{
    cout << "Start to train the network with files "
         << "\"" << path << "\" (cosmic) and "
         << "\"" << path2 << "\" (good events). "
         << endl;

    vector<double> input;
    input.reserve(modules.size());

    PRadDSTParser dst1, dst2;
    dst1.OpenInput(path);
    dst2.OpenInput(path2);

    int count = 0;
    double average = 0., average2 = 0.;
    PRadBenchMark timer;
    while(dst1.Read() && dst2.Read())
    {
        if((dst1.EventType() == PRadDSTParser::Type::event) &&
           (dst2.EventType() == PRadDSTParser::Type::event))
        {

            auto &cosmic = dst1.GetEvent();
            auto &product = dst2.GetEvent();

            if(!cosmic.is_physics_event() ||
               !product.is_physics_event())
            {
                continue;
            }

            if((++count)%PROGRESS_COUNT == 0) {
                cout << "----------event " << count
                     << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
                     << "\r" << flush;
            }

            FillInput(sys, input, cosmic);
            net.BP_Train(input, {1.});
            average += net.GetOutput().at(0);

            FillInput(sys, input, product);
            net.BP_Train(input, {0.});
            average2 += net.GetOutput().at(0);
        }
    }

    cout << "----------event " << count
         << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
         << endl;
    cout << "Finished training from file " << path << endl;
    cout << "Averaged output for cosmic is " << average/(double)count << endl;
    cout << "Averaged output for production is " << average2/(double)count << endl;
}

void FillInput(PRadHyCalSystem &sys, vector<double> &input, const EventData &event)
{
    sys.ChooseEvent(event);

    input.clear();
    for(auto &module : modules)
    {
        input.push_back(module->GetEnergy());
    }
}
