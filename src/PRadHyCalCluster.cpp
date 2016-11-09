//============================================================================//
// Basic PRad Cluster Reconstruction Class For HyCal                          //
// Different reconstruction methods can be implemented accordingly            //
//                                                                            //
// Chao Peng, Weizhi Xiong                                                    //
// 09/28/2016                                                                 //
//============================================================================//

#include <cmath>
#include <iostream>
#include <iomanip>
#include "PRadHyCalCluster.h"

PRadHyCalCluster::PRadHyCalCluster(PRadDataHandler *h)
: fHandler(h), fNHyCalClusters(0)
{
    // place holder
}

PRadHyCalCluster::~PRadHyCalCluster()
{
    // place holder
}

void PRadHyCalCluster::Clear()
{
    fNHyCalClusters = 0;
}

void PRadHyCalCluster::SetHandler(PRadDataHandler *h)
{
    fHandler = h;
}

void PRadHyCalCluster::UpdateModuleInfo()
{
    // to be implemented by methods
}

void PRadHyCalCluster::Reconstruct(EventData & /*event*/)
{
    // to be implemented by methods
}

// currently it only accepts GeV
void PRadHyCalCluster::NonLinearCorrection()
{
    for(int i = 0; i < fNHyCalClusters; ++i)
    {
        PRadDAQUnit *module = fHandler->GetChannelPrimex(fHyCalCluster[i].cid);
        float alpha = module->GetNonLinearConst();
        float Ecal = module->GetCalibrationEnergy();
        float ecorr = 1. + alpha*(fHyCalCluster[i].E - Ecal/1000.);
        // prevent unreasonably 
        if(fabs(ecorr - 1.) < 0.6)
            fHyCalCluster[i].E /= ecorr;
    }
}
