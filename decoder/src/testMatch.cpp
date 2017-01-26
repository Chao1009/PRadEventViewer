//============================================================================//
// An example showing how to reconstruct, transform and match the detectors   //
//                                                                            //
// Chao Peng                                                                  //
// 10/24/2016                                                                 //
//============================================================================//

#include "PRadDataHandler.h"
#include "PRadDSTParser.h"
#include "PRadInfoCenter.h"
#include "PRadBenchMark.h"
#include "PRadEPICSystem.h"
#include "PRadTaggerSystem.h"
#include "PRadHyCalSystem.h"
#include "PRadGEMSystem.h"
#include "PRadCoordSystem.h"
#include "PRadDetMatch.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(int /*argc*/, char * /*argv*/ [])
{

    PRadEPICSystem *epics = new PRadEPICSystem("config/epics_channels.conf");
    PRadHyCalSystem *hycal = new PRadHyCalSystem("config/hycal.conf");
    PRadGEMSystem *gem = new PRadGEMSystem("config/gem.conf");

    PRadDSTParser *dst_parser = new PRadDSTParser();
    dst_parser->EnableMode(PRadDSTParser::Mode::update_run_info);

    // coordinate system and detector match system
    PRadCoordSystem *coord_sys = new PRadCoordSystem("database/coordinates.dat");
    PRadDetMatch *det_match = new PRadDetMatch("config/det_match.conf");

    PRadBenchMark timer;

    //dst_parser->OpenInput("/work/hallb/prad/replay/prad_001288.dst");
    dst_parser->OpenInput("prad_1310.dst");
    hycal->ChooseRun("prad_1310.dst");
    std::cout << PRadInfoCenter::GetRunNumber() << std::endl;
    TFile f("prad_1310_match.root", "RECREATE");
    TH1F *hist[3];
    hist[0] = new TH1F("PbGlass R Diff", "Diff in R", 1000, -100, 100);
    hist[1] = new TH1F("PbWO4 R Diff", "Diff in R", 1000, -100, 100);
    hist[2] = new TH1F("Trans R Diff", "Diff in R", 1000, -100, 100);
    TH2F *hist2d = new TH2F("R Diff", "HyCal - GEM", 800, 0, 800, 200, -100, 100);

    PRadHyCalDetector *hycal_det = hycal->GetDetector();
    PRadGEMDetector *gem_det1 = gem->GetDetector("PRadGEM1");
    PRadGEMDetector *gem_det2 = gem->GetDetector("PRadGEM2");

    while(dst_parser->Read())
    {
        if(dst_parser->EventType() == PRadDSTParser::Type::event) {
            // you can push this event into data handler
            // handler->GetEventData().push_back(dst_parser->GetEvent()
            // or you can just do something with this event and discard it
            auto &event = dst_parser->GetEvent();

            // only interested in physics event
            if(!event.is_physics_event())
                continue;

            // update run information
            PRadInfoCenter::Instance().UpdateInfo(event);

            // reconstruct
            hycal->ChooseEvent(event);
            hycal->Reconstruct();
            gem->Reconstruct(event);

            // get reconstructed clusters
            auto &hycal_hit = hycal_det->GetHits();
            auto &gem1_hit = gem_det1->GetHits();
            auto &gem2_hit = gem_det2->GetHits();

            // coordinates transform, projection
            coord_sys->Transform(PRadDetector::HyCal, hycal_hit.begin(), hycal_hit.end());
            coord_sys->Transform(PRadDetector::PRadGEM1, gem1_hit.begin(), gem1_hit.end());
            coord_sys->Transform(PRadDetector::PRadGEM2, gem2_hit.begin(), gem2_hit.end());

            coord_sys->Projection(hycal_hit.begin(), hycal_hit.end());
            coord_sys->Projection(gem1_hit.begin(), gem1_hit.end());
            coord_sys->Projection(gem2_hit.begin(), gem2_hit.end());

            // hits matching, return matched index
            auto matched = det_match->Match(hycal_hit, gem1_hit, gem2_hit);

            for(auto idx : matched)
            {
                auto &hit = hycal_hit[idx.hycal];

                int hidx = 0;
                if(TEST_BIT(hit.flag, kPbWO4))
                    hidx = 1;
                if(TEST_BIT(hit.flag, kTransition))
                    hidx = 2;

                float r = sqrt(hit.x*hit.x + hit.y*hit.y);
                float x, y;
                if(idx.gem1 >= 0) {
                    x = gem1_hit[idx.gem1].x;
                    y = gem1_hit[idx.gem1].y;
                } else {
                    x = gem2_hit[idx.gem2].x;
                    y = gem2_hit[idx.gem2].y;
                }

                float dr = r - sqrt(x*x + y*y);
                hist[hidx]->Fill(dr);
                hist2d->Fill(r, dr);
            }

        } else if(dst_parser->EventType() == PRadDSTParser::Type::epics) {
            // save epics into handler, otherwise get epicsvalue won't work
            epics->AddEvent(dst_parser->GetEPICSEvent());
        }
    }

    dst_parser->CloseInput();

    cout << "TIMER: Finished, took " << timer.GetElapsedTime() << " ms" << endl;
    cout << PRadInfoCenter::GetBeamCharge() << endl;
    cout << PRadInfoCenter::GetLiveTime() << endl;

    for(int i = 0; i < 3; ++i)
    {
        hist[i]->Write();
    }
    hist2d->Write();
    f.Close();
//    handler->WriteToDST("prad_001323_0-10.dst");
    //handler->PrintOutEPICS();
    return 0;
}
