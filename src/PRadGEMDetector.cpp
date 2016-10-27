//============================================================================//
// GEM detector class                                                         //
// A detector has several planes (X, Y)                                       //
//                                                                            //
// Chao Peng                                                                  //
// 10/07/2016                                                                 //
//============================================================================//

#include "PRadGEMDetector.h"
#include "PRadGEMSystem.h"
#include "PRadGEMCluster.h"
#include "PRadGEMAPV.h"
#include <algorithm>
#include <iostream>

PRadGEMDetector::PRadGEMDetector(PRadGEMSystem *g,
                                 const std::string &readoutBoard,
                                 const std::string &detectorType,
                                 const std::string &detector)
: gem_srs(g), name(detector), type(detectorType), readout_board(readoutBoard),
  NClusters(0)
{
    planes.resize(PRadGEMPlane::Plane_Max, nullptr);
    det_id = PRadDetectors::getID(name.c_str());
}

PRadGEMDetector::~PRadGEMDetector()
{
    for(auto &plane : planes)
    {
        if(plane != nullptr)
            delete plane, plane = nullptr;
    }
}

void PRadGEMDetector::AddPlane(const PRadGEMPlane::PlaneType &type,
                               const std::string &name,
                               const double &size,
                               const int &conn,
                               const int &ori,
                               const int &dir)
{
    planes[(int)type] = new PRadGEMPlane(this, name, type, size, conn, ori, dir);
}

void PRadGEMDetector::AddPlane(const int &type, PRadGEMPlane *plane)
{
    if(plane->GetDetector() != nullptr) {
        std::cerr << "PRad GEM Detector Error: "
                  << "Trying to add plane " << plane->GetName()
                  << " to detector " << name
                  << ", but the plane is belong to " << plane->GetDetector()->name
                  << std::endl;
        return;
    }

    if(planes[type] != nullptr) {
        std::cout << "PRad GEM Detector Warning: "
                  << "Trying to add multiple planes with the same type "
                  << "to detector " << name
                  << ", there will be potential memory leakage if the original "
                  << "plane is not released properly."
                  << std::endl;
    }

    plane->SetDetector(this);
    planes[type] = plane;
}

void PRadGEMDetector::ReconstructHits(PRadGEMCluster *gem_recon)
{
    for(auto &plane : planes)
    {
        if(plane == nullptr)
            continue;
        gem_recon->Reconstruct(plane);
    }

    NClusters = gem_recon->FormClusters(this);
}

void PRadGEMDetector::ReconstructHits()
{
    PRadGEMCluster *gem_recon = gem_srs->GetClusterMethod();
    ReconstructHits(gem_recon);
}

void PRadGEMDetector::ClearHits()
{
    for(auto &plane : planes)
    {
        if(plane != nullptr)
            plane->ClearPlaneHits();
    }

    NClusters = 0;
}

void PRadGEMDetector::AssignID(const int &i)
{
    id = i;
}

std::vector<PRadGEMPlane*> PRadGEMDetector::GetPlaneList()
{
    // since it allows nullptr in planes
    // for safety issue, only pack existing planes and return
    std::vector<PRadGEMPlane*> result;

    for(auto &plane : planes)
    {
        if(plane != nullptr)
            result.push_back(plane);
    }

    return result;
}

PRadGEMPlane *PRadGEMDetector::GetPlane(const int &type)
{
    return planes[type];
}

std::vector<PRadGEMAPV*> PRadGEMDetector::GetAPVList(const int &type)
{
    if(planes[type] == nullptr)
        return std::vector<PRadGEMAPV*>();

    return planes[type]->GetAPVList();
}

std::list<GEMPlaneCluster> &PRadGEMDetector::GetPlaneClusters(const int &type)
throw (PRadException)
{
    if(planes[type] == nullptr)
        throw PRadException("PRadGEMDetector Error", "Plane does not exist!");

    return planes[type]->GetPlaneClusters();
}

std::vector<std::list<GEMPlaneCluster>*> PRadGEMDetector::GetDetectorClusters()
{
    std::vector<std::list<GEMPlaneCluster>*> plane_clusters;

    for(auto &plane : planes)
    {
        if(plane != nullptr)
            plane_clusters.push_back(&plane->GetPlaneClusters());
        else
            plane_clusters.push_back(nullptr);
    }

    return plane_clusters;
}

void PRadGEMDetector::ConnectAPV(const int &type, PRadGEMAPV *apv, const int &index)
{
    if(planes[type] == nullptr)
        return;

    planes[type]->ConnectAPV(apv, index);
}

GEMHit *PRadGEMDetector::GetCluster(int &n)
{
    n = NClusters;
    return gem_clusters;
}

std::vector<GEMHit> PRadGEMDetector::GetCluster()
{
    std::vector<GEMHit> result;

    for(int i = 0; i < NClusters; ++i)
    {
        result.push_back(gem_clusters[i]);
    }

    return result;
}
