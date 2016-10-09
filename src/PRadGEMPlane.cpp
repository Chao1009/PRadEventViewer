//============================================================================//
// GEM plane class                                                            //
// A GEM plane is a component of GEM detector                                 //
// It is connected to several APV units                                       //
// GEM hits are collected on plane level and reconstructed in cluster         //
//                                                                            //
// Chao Peng                                                                  //
// 10/07/2016                                                                 //
//============================================================================//

#include "PRadGEMPlane.h"
#include "PRadGEMAPV.h"
#include <iostream>
#include <algorithm>


PRadGEMPlane::PRadGEMPlane()
: detector(nullptr), name("Undefined"), type(Plane_X), size(0.),
  connector(-1), orientation(0)
{
}

PRadGEMPlane::PRadGEMPlane(const std::string &n, const PlaneType &t, const double &s,
                           const int &c, const int &o)
: detector(nullptr), name(n), type(t), size(s), connector(c), orientation(o)
{
    apv_list.resize(c, nullptr);
}

PRadGEMPlane::PRadGEMPlane(PRadGEMDetector *d, const std::string &n, const PlaneType &t,
                           const double &s, const int &c, const int &o)
: detector(d), name(n), type(t), size(s), connector(c), orientation(o)
{
    apv_list.resize(c, nullptr);
}

PRadGEMPlane::~PRadGEMPlane()
{
    for(auto &apv : apv_list)
    {
        if(apv != nullptr)
            apv->SetDetectorPlane(nullptr);
    }
}

void PRadGEMPlane::SetCapacity(const int &c)
{
    if(c < connector)
    {
        std::cout << "PRad GEM Plane Warning: Reduce the connectors on plane "
                  << name << " from " << connector << " to " << c
                  << ". Thus it will lose the connection between APVs that beyond "
                  << c
                  << std::endl;
    }
    connector = c;

    apv_list.resize(c, nullptr);
}

void PRadGEMPlane::ConnectAPV(PRadGEMAPV *apv)
{
    if((size_t)apv->GetPlaneIndex() >= apv_list.size()) {
        std::cout << "PRad GEM Plane Warning: Failed to connect plane " << name
                  << " with APV " << apv->GetAddress()
                  << ". Plane connectors are not enough, have " << connector
                  << ", this APV is at " << apv->GetPlaneIndex()
                  << std::endl;
        return;
    }

    if(apv_list[apv->GetPlaneIndex()] != nullptr) {
        std::cout << "PRad GEM Plane Warning: The connector " << apv->GetPlaneIndex()
                  << " of plane " << name << " is connected to APV " << apv->GetAddress()
                  << ", replace the connection."
                  << std::endl;
        return;
    }

    apv_list[apv->GetPlaneIndex()] = apv;
    apv->SetDetectorPlane(this);
}

void PRadGEMPlane::DisconnectAPV(const size_t &plane_index)
{
    if(plane_index >= apv_list.size())
        return;

    apv_list[plane_index] = nullptr;
}

std::vector<PRadGEMAPV*> PRadGEMPlane::GetAPVList()
{
    // since the apv list may contain nullptr,
    // only pack connected APVs and return
    std::vector<PRadGEMAPV*> result;

    for(const auto &apv : apv_list)
    {
        if(apv != nullptr)
            result.push_back(apv);
    }

    return result;
}

// unit is mm
double PRadGEMPlane::GetStripPosition(const int &plane_strip)
{
    double pitch = 0.4;
    if(type == Plane_X)
        return -0.5*(size + 31*pitch) + pitch*plane_strip;
    else
        return -0.5*(size - pitch) + pitch*plane_strip;
}

double PRadGEMPlane::GetMaxCharge(const std::vector<float> &charges)
{
    if(!charges.size())
        return 0.;

    double result = charges.at(0);

    for(size_t i = 1; i < charges.size(); ++i)
    {
        if(result < charges.at(i))
            result = charges.at(i);
    }

    return result;
}

double PRadGEMPlane::GetIntegratedCharge(const std::vector<float> &charges)
{
    double result = 0.;

    for(auto &charge : charges)
        result += charge;

    return result;
}

void PRadGEMPlane::ClearPlaneHits()
{
    hit_list.clear();
}

void PRadGEMPlane::AddPlaneHit(const int &plane_strip, const std::vector<float> &charges)
{
    // X plane needs to remove 16 strips at both ends
    if((type == Plane_X) &&
       ((plane_strip < 16) || (plane_strip > 1391)))
       return;

    hit_list.emplace_back(plane_strip, GetMaxCharge(charges));
}

void PRadGEMPlane::CollectAPVHits()
{
    ClearPlaneHits();

    for(auto &apv : apv_list)
    {
        if(apv != nullptr)
            apv->CollectZeroSupHits();
    }
}

void PRadGEMPlane::ClusterHits()
{
    // clear plane clusters
    cluster_list.clear();

    // sort the hits by its strip number
    std::sort(hit_list.begin(), hit_list.end(),
              // lamda expr, compare hit by their strip numbers
              [](const GEMPlaneHit &h1, const GEMPlaneHit &h2)
              {
                  return h1.strip < h2.strip;
              });

    // group the hits that have consecutive strip number
    auto cluster_begin = hit_list.begin();
    for(auto it = hit_list.begin(); it != hit_list.end(); ++it)
    {
        auto next = it + 1;

        // end of list, group the last cluster
        if(next == hit_list.end()) {
            cluster_list.emplace_back(std::vector<GEMPlaneHit>(cluster_begin, next));
            break;
        }

        // check consecutivity
        if(next->strip - it->strip > 1) {
            cluster_list.emplace_back(std::vector<GEMPlaneHit>(cluster_begin, next));
            cluster_begin = next;
        }
    }

    // split the clusters that may contain multiple physical hits
    splitCluster();

    // remove the clusters that does not pass certain criteria
    filterCluster();
}

void PRadGEMPlane::splitCluster()
{
    // We are trying to find the valley that satisfies certain critieria,
    // i.e., less than a sigma comparing to neighbor strips on both sides.
    // Such valley probably means two clusters are grouped together, so it
    // will be separated, and each gets 1/2 of the charge from the overlap
    // strip.

    // loop over the cluster list
    for(auto c_it = cluster_list.begin(); c_it != cluster_list.end(); ++c_it)
    {
        auto &hits = c_it->hits;

        // no need to do separation if less than 3 hits
        if(hits.size() < 3)
            continue;

        // we use 3 consecutive iterator
        auto it_prev = hits.begin();
        auto it = it_prev + 1;
        auto it_next = it + 1;
        for(; it_next != hits.end(); ++it_prev, ++it, ++it_next)
        {
            if( (it_prev->charge - it->charge > 14) &&
                (it_next->charge - it->charge > 14) )
            {
                // split charge
                it->charge /= 2.;

                // create new cluster, and insert behind current iterator
                GEMPlaneCluster split(std::vector<GEMPlaneHit>(it, hits.end()));
                cluster_list.insert(std::next(c_it, 1), split);

                // remove the hits that are grouped into a new cluster
                // keep the overlap strip
                hits.erase(it_next, hits.end());

                // no need to continue, since the left hits are moved into next
                // cluster
                break;
            }
        }
    }
}

bool PRadGEMPlane::inspectCluster(const GEMPlaneCluster &c)
{
    //TODO make parameters configurable

    // only check size for now
    if(c.hits.size() < 1 || c.hits.size() > 20)
        return false;

    // passed all the check
    return true;
}

void PRadGEMPlane::filterCluster()
{
    for(auto it = cluster_list.begin(); it != cluster_list.end(); ++it)
    {
        // carefully remove element inside the loop
        if(!inspectCluster(*it))
            cluster_list.erase(it--);
    }
}

