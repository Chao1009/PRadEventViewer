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
#include <random>
#include <vector>

#define PROGRESS_COUNT 1000

using namespace std;

void NeuralTrain(CNeuralNetwork &net,
                 PRadHyCalSystem &sys,
                 const string &path,
                 const string &path2,
                 int train_times,
                 unsigned int cap);

void FillParams(PRadHyCalSystem &sys,
                const string &path,
                vector<vector<double>> &params,
                unsigned int cap);

void Helper(const string &mes = "")
{
    if(mes.size())
        cerr << "error: " << mes << endl;

    cout << "usage: " << endl
         << setw(10) << " " << "neuralTrain <cosmic_data> <good_data>"
         << endl
         << "options:" << endl
         << setw(10) << "-n " << "<file_path>: "
         << "create network from file, a new network will be created by default."
         << endl
         << setw(10) << "-l " << "<\"value1, value2, ...\">: "
         << "set the hidden layers for neural network, {20, 10, 5, 3} is the default value."
         << endl
         << setw(10) << "-s " << "<file_path>: "
         << "set the path to save the trained network, save to \"saved.net\" by default "
         << endl
         << setw(10) << "-f " << "<value>: "
         << "define learning factor, 0.1 is the default value."
         << endl
         << setw(10) << "-t " << "<value>: "
         << "set the training times (1,000 as the unit), 5,000k is the default value."
         << endl
         << setw(10) << "-c " << "<value>: "
         << "set the training bank capacity (1,000 as the unit), 1,000k is the default value."
         << endl
         << setw(10) << "-h " << ": "
         << "see this helper information."
         << endl;

    exit(-1);
}

int main(int argc, char *argv[])
{
    string net_path, save_path, layer_str, argstr[2];
    double learn_factor = 0.1;
    int learn_times = 5000, cap = 1000, run_arg = 0;

    save_path = "saved.net";

    for(int i = 1; i < argc; ++i)
    {
        char* ptr = argv[i];
        if(*(ptr++) == '-')
        {
            string option;
            option += *(ptr++);

            switch(option.at(0))
            {
            case 'n':
                if(i >= argc - 1)
                    Helper("incorrect format for option " + option);
                net_path = argv[++i];
                break;
            case 's':
                if(i >= argc - 1)
                    Helper("incorrect format for option " + option);
                save_path = argv[++i];
                break;
            case 'f':
                if(i >= argc - 1)
                    Helper("incorrect format for option " + option);
                learn_factor = strtod(argv[++i], nullptr);
                break;
            case 't':
                if(i >= argc - 1)
                    Helper("incorrect format for option " + option);
                learn_times = atoi(argv[++i]);
                break;
            case 'c':
                if(i >= argc - 1)
                    Helper("incorrect format for option " + option);
                cap = atoi(argv[++i]);
                break;
            case 'l':
                if(i >= argc - 1)
                    Helper("incorrect format for option " + option);
                layer_str = argv[++i];
                break;
            case 'h':
            default:
                Helper("unknown option " + option);
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
        Helper("unmatched number of inputs, exepcting 2.");
    }

    string cosmic_file = argstr[0];
    string good_file = argstr[1];
    CNeuralNetwork my_net(learn_factor);

    vector<unsigned int> hidden;
    if(layer_str.empty())
    {
        // 4 hidden layers with 20, 10, 5, 3 neurons
        hidden = {20, 10, 5, 3};
    }
    else
    {
        auto vals = ConfigParser::split(layer_str, ",");
        while(vals.size())
        {
            string val = move(vals.front());
            vals.pop();
            hidden.push_back(stoi(val));
        }
    }

    if(net_path.empty())
    {
        // create net with dimensions
        // inputs and outputs are hard coded to be 6 and 1
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

    NeuralTrain(my_net, sys, cosmic_file, good_file, learn_times*1000, cap*1000);

    my_net.SaveNet(save_path.c_str());
    return 0;
}

void NeuralTrain(CNeuralNetwork &net,
                 PRadHyCalSystem &sys,
                 const string &path,
                 const string &path2,
                 int number,
                 unsigned int cap)
{

    vector<vector<double>> cosmic_params;
    vector<vector<double>> good_params;

    // read events and fill in the params array
    FillParams(sys, path, cosmic_params, cap);
    FillParams(sys, path2, good_params, cap);

    cout << "Start to train the network with files "
         << endl
         << "\"" << path << "\" (cosmic)."
         << endl
         << "\"" << path2 << "\"(good events)."
         << endl;

    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<unsigned int> cosmic_dist(0, cosmic_params.size()-1);
    std::uniform_int_distribution<unsigned int> good_dist(0, good_params.size()-1);

    int count = 0;
    PRadBenchMark timer;
    while(++count <= number)
    {
        if(count%PROGRESS_COUNT == 0) {
            cout << "----------training " << count
                 << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
                 << "\r" << flush;
        }
        // randomly pick param sets
        auto &cosmic_input = cosmic_params.at(cosmic_dist(rng));
        net.BP_Train(cosmic_input, {1.0});

        auto &good_input = good_params.at(good_dist(rng));
        net.BP_Train(good_input, {0.0});
    }

    cout << "----------training " << count - 1
         << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
         << endl;
    cout << "Training finished, now test the result." << endl;

    // check the training result
    count = 0;
    double average1 = 0., average2 = 0.;
    while(++count <= 10000)
    {
        auto &cosmic_input = cosmic_params.at(cosmic_dist(rng));
        net.Update(cosmic_input);
        average1 += net.GetOutput().at(0);

        auto &good_input = good_params.at(good_dist(rng));
        net.Update(good_input);
        average2 += net.GetOutput().at(0);
    }

    cout << "The average output for cosmic is " << average1/(double)count
         << endl
         << "The average output for good event is " << average2/(double)count
         << endl;
}

void FillParams(PRadHyCalSystem &sys,
                const string &path,
                vector<vector<double>> &params,
                unsigned int cap)
{

    PRadDSTParser dst;
    dst.OpenInput(path);

    int count = 0;
    PRadBenchMark timer;

    cout << "Prepare inputs from file "
         << "\"" << path << "\" (Capacity: " << cap << ")"
         << endl;

    params.clear();
    params.reserve(cap);

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

            params.emplace_back(move(param.GetParamList()));

            if(params.size() >= cap)
                break;
        }
    }

    cout << "----------event " << count
         << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
         << endl;
}
