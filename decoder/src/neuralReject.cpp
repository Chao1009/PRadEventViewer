//============================================================================//
// An example showing how to use the neural network to reject cosmic.         //
// The neural network should be trained first.                                //
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
#include "TFile.h"
#include "TH1.h"

//#define NEW_NET
#define PROGRESS_COUNT 10000

using namespace std;

void NeuralReject(CNeuralNetwork &net, PRadHyCalSystem &sys, string path);

int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        cout << "usage: neuralReject <net_file> <data_file1> <data_file2>..."
             << endl;
        return 0;
    }

    string net_file = argv[1];
    CNeuralNetwork my_net;
    my_net.CreateNet(net_file.c_str());

    PRadHyCalSystem sys;
    sys.Configure("config/hycal.conf");

    for(int i = 2; i < argc; ++i)
    {
        string file = argv[i];
        NeuralReject(my_net, sys, file);
    }

    return 0;
}

void NeuralReject(CNeuralNetwork &net, PRadHyCalSystem &sys, string path)
{
    string fname = ConfigParser::decompose_path(path).name;

    TFile f((fname + "_prob.root").c_str(), "RECREATE");
    TH1F hist("Cosmic Probability", "Cosmic Probability", 100, 0., 1.0);

    vector<PRadHyCalModule*> modules = sys.GetDetector()->GetModuleList();

    sort(modules.begin(), modules.end(),
         [] (PRadHyCalModule *m1, PRadHyCalModule *m2)
         {
            if(m1->GetY() == m2->GetY())
                return m1->GetX() < m2->GetX();
            else
                return m1->GetY() < m2->GetY();
         });

    vector<double> input;
    input.reserve(modules.size());

    PRadDSTParser dst_parser;
    dst_parser.OpenInput(path);

    int count = 0, reject = 0;
    PRadBenchMark timer;
    while(dst_parser.Read())
    {
        if(dst_parser.EventType() == PRadDSTParser::Type::event) {

            auto &event = dst_parser.GetEvent();

            if(!event.is_physics_event())
                continue;

            if((++count)%PROGRESS_COUNT == 0) {
                cout << "----------event " << count
                     << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
                     << "\r" << flush;
            }

            // choose this event, get energies for all modules
            sys.ChooseEvent(event);
            input.clear();
            for(auto &module : modules)
            {
                input.push_back(module->GetEnergy());
            }
            net.Update(input);
            hist.Fill(net.GetOutput().at(0));
            if(net.GetOutput().at(0) > 0.9)
                reject++;
        }
    }

    cout << "----------event " << count
         << "-------[ " << timer.GetElapsedTimeStr() << " ]------"
         << endl;
    cout << "Finished cosmic rejection for file " << path << endl;
    cout << "Read " << count << " events, rejected " << reject << "." << endl;

    hist.Write();
    f.Save();
}
