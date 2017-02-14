//============================================================================//
// Match the HyCal Clusters and GEM clusters                                  //
// The clusters are required to be in the same frame (beam center frame)      //
// The frame transform work can be done in PRadCoordSystem                    //
//                                                                            //
// Details on projection:                                                     //
// By calling PRadCoordSystem::ProjectionDistance, the z from two clusters    //
// are compared first, if they are not at the same XOY-plane, the one with    //
// smaller z will be projected to the larger z. (from target point (0, 0, 0)) //
// If do not want the automatically projection, be sure you have projected    //
// the clusters to the same XOY-plane first.                                  //
//                                                                            //
//                                                                            //
// Xinzhan Bai, first version                                                 //
// Weizhi Xiong, adapted to PRad analysis software                            //
// Chao Peng, removed coordinates manipulation, only left matching part       //
// 10/22/2016                                                                 //
//============================================================================//

#include "PRadDetMatch.h"
#include "PRadCoordSystem.h"

// constructor
PRadDetMatch::PRadDetMatch(const std::string &path)
{
    Configure(path);
}

// destructor
PRadDetMatch::~PRadDetMatch()
{
    // place holder
}

void PRadDetMatch::Configure(const std::string &path)
{
    // if no configuration file specified, load the default value quietly
    bool verbose = false;

    if(!path.empty()) {
        ConfigObject::Configure(path);
        verbose = true;
    }

    leadGlassRes = getDefConfig<float>("Lead_Glass_Resolution", 10, verbose);
    transitionRes = getDefConfig<float>("Transition_Resolution", 7, verbose);
    crystalRes = getDefConfig<float>("Crystal_Resolution", 3, verbose);
    gemRes = getDefConfig<float>("GEM_Resolution", 0.08, verbose);
    matchSigma = getDefConfig<float>("Match_Factor", 5, verbose);
    overlapSigma = getDefConfig<float>("GEM_Overlap_Factor", 10, verbose);
}

std::vector<MatchedIndex> PRadDetMatch::Match(std::vector<HyCalHit> &hycal,
                                              std::vector<GEMHit> &gem1,
                                              std::vector<GEMHit> &gem2)
const
{
    std::vector<MatchedIndex> result;

    for(size_t i = 0; i < hycal.size(); ++i)
    {
        MatchedIndex index(i);

        // pre match, only check if distance is within the range
        // fill in hits as candidates
        for(size_t j = 0; j < gem1.size(); ++j)
        {
            if(PreMatch(hycal[i], gem1[j]))
                index.gem1_cand.push_back(j);
        }
        for(size_t j = 0; j < gem2.size(); ++j)
        {
            if(PreMatch(hycal[i], gem2[j]))
                index.gem2_cand.push_back(j);
        }

        // post match, do further check, which one is the closest
        // and if the two gem hits are overlapped
        if(PostMatch(index, hycal[i], &gem1[0], &gem2[0]))
            result.push_back(index);
    }

    return result;
}

// project 1 HyCal cluster and 1 GEM cluster to HyCal Plane.
// if they are at the same z, there will be no projection, otherwise they are
// projected to the furthest z
// return true if they are within certain range (depends on HyCal resolution)
// return false if they are not
bool PRadDetMatch::PreMatch(const HyCalHit &hycal, const GEMHit &gem)
const
{
    // lead glass (largest) value as default
    float base_range = leadGlassRes;
    // crystal region
    if(TEST_BIT(hycal.flag, kPbWO4))
        base_range = crystalRes;
    // transition region
    if(TEST_BIT(hycal.flag, kTransition))
        base_range = transitionRes;

    float dist = PRadCoordSystem::ProjectionDistance(hycal, gem);

    if(dist > matchSigma * base_range)
        return false;
    else
        return true;
}

bool PRadDetMatch::PostMatch(MatchedIndex &index,
                             HyCalHit &hycal, GEMHit *gem1, GEMHit *gem2)
const
{
    // no candidates
    if(index.gem1_cand.empty() && index.gem2_cand.empty())
        return false;

    // find the closest hits from gem1 and gem2
    // large initial value
    float dist = 1e3;
    for(auto &idx : index.gem1_cand)
    {
        float new_dist = PRadCoordSystem::ProjectionDistance(hycal, gem1[idx]);
        if(new_dist < dist) {
            dist = new_dist;
            index.gem1 = idx;
        }
    }

    dist = 1e3;
    for(auto &idx : index.gem2_cand)
    {
        float new_dist = PRadCoordSystem::ProjectionDistance(hycal, gem2[idx]);
        if(new_dist < dist) {
            dist = new_dist;
            index.gem2 = idx;
        }
    }

    if(index.gem1 >= 0)
        SET_BIT(hycal.flag, kGEM1Match);
    if(index.gem2 >= 0)
        SET_BIT(hycal.flag, kGEM2Match);

    // both gem1 and gem2 have matched hits, check if they are overlapped
    if(index.gem1 >= 0 && index.gem2 >= 0) {
        float gem_dist = PRadCoordSystem::ProjectionDistance(gem1[index.gem1], gem2[index.gem2]);
        // they are not overlap hits
        if(gem_dist > overlapSigma * gemRes) {
            float gem1_dist = PRadCoordSystem::ProjectionDistance(gem1[index.gem1], hycal);
            float gem2_dist = PRadCoordSystem::ProjectionDistance(gem2[index.gem2], hycal);
            // gem2 has a better match
            if(gem2_dist < gem1_dist) {
                index.gem1 = -1;
                CLEAR_BIT(hycal.flag, kGEM1Match);
            } else {
                index.gem2 = -1;
                CLEAR_BIT(hycal.flag, kGEM2Match);
            }
        }
    }

    return true;
}
