//============================================================================//
// GEM plane class                                                            //
// A GEM plane is a component of GEM detector, it should never exist without  //
// a detector, thus the memory of plane will be managed in GEM detector       //
//                                                                            //
// A GEM plane can be connected to several APV units                          //
// GEM hits are collected and grouped on plane level                          //
//                                                                            //
// Chao Peng, Frame work of this class                                        //
// Xinzhan Bai, position and charge calculation method                        //
// 10/07/2016                                                                 //
//============================================================================//

#include "PRadGEMPlane.h"
#include "PRadGEMAPV.h"
#include <iostream>
#include <iterator>
#include <algorithm>



//============================================================================//
// constructor, assigment operator, destructor                                //
//============================================================================//

// constructor
PRadGEMPlane::PRadGEMPlane(PRadGEMDetector *det)
: detector(det), name("Undefined"), type(Plane_X), size(0.), orient(0)
{
    // place holder
}

// constructor
PRadGEMPlane::PRadGEMPlane(const std::string &n, const PlaneType &t, const float &s,
                           const int &c, const int &o, const int &d, PRadGEMDetector *det)
: detector(det), name(n), type(t), size(s), orient(o), direction(d)
{
    apv_list.resize(c, nullptr);
}

// copy constructor
// connections between it and apv/detector won't be copied
PRadGEMPlane::PRadGEMPlane(const PRadGEMPlane &that)
: name(that.name), type(that.type), size(that.size), orient(that.orient),
  direction(that.direction), hit_list(that.hit_list), cluster_list(that.cluster_list)
{
    apv_list.resize(that.apv_list.size(), nullptr);
}

// move constructor
PRadGEMPlane::PRadGEMPlane(PRadGEMPlane &&that)
: name(std::move(that.name)), type(that.type), size(that.size), orient(that.orient),
  direction(that.direction), hit_list(std::move(hit_list)), cluster_list(std::move(that.cluster_list))
{
    apv_list.resize(that.apv_list.size(), nullptr);
}

// destructor
PRadGEMPlane::~PRadGEMPlane()
{
    ResetConnections();
}

// copy assignment
PRadGEMPlane &PRadGEMPlane::operator =(const PRadGEMPlane &rhs)
{
    PRadGEMPlane that(rhs);
    *this = std::move(that);
    return *this;
}

PRadGEMPlane &PRadGEMPlane::operator =(PRadGEMPlane &&rhs)
{
    // clear all the connections
    detector = nullptr;
    ResetConnections();

    name = std::move(rhs.name);
    type = rhs.type;
    size = rhs.size;
    orient = rhs.orient;
    direction = rhs.direction;

    hit_list = std::move(rhs.hit_list);
    cluster_list = std::move(rhs.cluster_list);

    apv_list.resize(rhs.apv_list.size(), nullptr);
    return *this;
}



//============================================================================//
// Public Member Functions                                                    //
//============================================================================//

// change the capacity
void PRadGEMPlane::SetCapacity(int c)
{
    // capacity cannot be negative
    if(c < 0) c = 0;

    if((size_t)c < apv_list.size())
    {
        std::cout << "PRad GEM Plane Warning: Reduce the connectors on plane "
                  << name << " from " << apv_list.size() << " to " << c
                  << ". Thus it will lose the connection between APVs that beyond "
                  << c
                  << std::endl;

        for(size_t i = c; i < apv_list.size(); ++i)
        {
            if(apv_list[i] != nullptr)
                apv_list[i]->DisconnectPlane();
        }
    }

    apv_list.resize(c, nullptr);
}

// connect an APV to the plane
void PRadGEMPlane::ConnectAPV(PRadGEMAPV *apv, const int &index)
{
    if(apv == nullptr)
        return;

    if((size_t)index >= apv_list.size()) {
        std::cout << "PRad GEM Plane Warning: Failed to connect plane " << name
                  << " with APV " << apv->GetAddress()
                  << ". Plane connectors are not enough, have " << apv_list.size()
                  << ", this APV is to be connected at " << index
                  << std::endl;
        return;
    }

    if(apv_list[index] != nullptr) {
        std::cout << "PRad GEM Plane Warning: The connector " << index
                  << " of plane " << name << " is connected to APV " << apv->GetAddress()
                  << ", replace the connection."
                  << std::endl;
        return;
    }

    apv_list[index] = apv;
    apv->SetDetectorPlane(this, index);
}

// disconnect an APV
void PRadGEMPlane::DisconnectAPV(const size_t &plane_index)
{
    if(plane_index >= apv_list.size())
        return;

    apv_list[plane_index] = nullptr;
}

// reset all APV connections
void PRadGEMPlane::ResetConnections()
{
    for(auto &apv : apv_list)
    {
        if(apv != nullptr)
            apv->DisconnectPlane();
    }
}

// get existing APV list
std::vector<PRadGEMAPV*> PRadGEMPlane::GetAPVList()
const
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

// calculate strip position by plane strip index
float PRadGEMPlane::GetStripPosition(const int &plane_strip)
const
{
    float position;

    if(type == Plane_X) {
        position = -0.5*(size + 31*STRIP_PITCH) + STRIP_PITCH*plane_strip - X_SHIFT;
    } else {
        position = -0.5*(size - STRIP_PITCH) + STRIP_PITCH*plane_strip;
    }

    return direction*position;
}

// get the maximum charge from input charges
float PRadGEMPlane::GetMaxCharge(const std::vector<float> &charges)
const
{
    if(!charges.size())
        return 0.;

    float result = charges.at(0);

    for(size_t i = 1; i < charges.size(); ++i)
    {
        if(result < charges.at(i))
            result = charges.at(i);
    }

    return result;
}

// get the average charge from input charges
float PRadGEMPlane::GetIntegratedCharge(const std::vector<float> &charges)
const
{
    float result = 0.;

    for(auto &charge : charges)
        result += charge;

    return result;
}

// clear the stored plane hits
void PRadGEMPlane::ClearPlaneHits()
{
    hit_list.clear();
}

// add a plane hit
// X plane needs to remove 16 strips at both ends, because they are floating
// This is a special setup for PRad GEMs, so not configurable
void PRadGEMPlane::AddPlaneHit(const int &plane_strip, const std::vector<float> &charges)
{
    if((type == Plane_X) &&
       ((plane_strip < 16) || (plane_strip > 1391)))
       return;

    hit_list.emplace_back(plane_strip, GetMaxCharge(charges));
}

// collect hits from the connected APVs
void PRadGEMPlane::CollectAPVHits()
{
    ClearPlaneHits();

    for(auto &apv : apv_list)
    {
        if(apv != nullptr)
            apv->CollectZeroSupHits();
    }
}

