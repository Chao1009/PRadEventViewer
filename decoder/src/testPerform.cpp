//============================================================================//
// An example to test the performance of reconstruction methods               //
//                                                                            //
// Chao Peng                                                                  //
// 11/12/2016                                                                 //
//============================================================================//

#include "PRadHyCalSystem.h"
#include "PRadDataHandler.h"
#include "PRadDSTParser.h"
#include "PRadBenchMark.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#define PROGRESS_COUNT 10000

using namespace std;

void testHyCalCluster(const string &file, PRadHyCalSystem *sys);

int main(int argc, char *argv[])
{
    if(argc < 2) {
        cout << "usage: testPerform <file1> <file2> ..." << endl;
        return 0;
    }

    PRadHyCalSystem *hycal_sys = new PRadHyCalSystem("config/hycal.conf");

    for(int i = 1; i < argc; ++i)
    {
        string file = argv[i];
        testHyCalCluster(file, hycal_sys);
    }

    return 0;
}

void testHyCalCluster(const string &file, PRadHyCalSystem *sys)
{
    PRadDSTParser *dst_parser = new PRadDSTParser();

    cout << "Test HyCal Clustering Performance for file " << file << endl;
    cout << "Using method " << sys->GetClusterMethodName() << endl;

    dst_parser->OpenInput(file);

    PRadBenchMark timer;

    int count = 0;
    double time = 0;
    while(dst_parser->Read())
    {
        if(dst_parser->EventType() == PRadDSTParser::Type::event) {

            auto event = dst_parser->GetEvent();
            if(!event.is_physics_event())
                continue;

            ++count;

            if(count%PROGRESS_COUNT == 0) {
                double t = timer.GetElapsedTime();
                time += t;
                timer.Reset();

                cout <<"------[ ev " << count << " ]---"
                     << "---[ " << t << " ms ]---"
                     << "---[ " << time/(double)count << " ms/ev ]------"
                     << "\r" << flush;
            }
            sys->Reconstruct(event);
        }
    }

    dst_parser->CloseInput();
    time += timer.GetElapsedTime();
    cout << endl;
    cout << "TIMER: Finished, read and reconstructed " << count << " events, "
         << "using method " << sys->GetClusterMethodName() << ", "
         << "took " << time/1000. << " s."
         << endl;
}
