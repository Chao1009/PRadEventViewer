//========================================================//
//match the HyCal Clusters and GEM clusters, both of them //
//need to be transformed to the Lab frame in advance      //
//========================================================//

#ifndef PRAD_DET_MATCH_H
#define PRAD_DET_MATCH_H

#include <string>
#include <vector>
#include "PRadEventStruct.h"
#include "ConfigObject.h"


struct MatchedIndex
{
    int hycal;
    int gem1;
    int gem2;
    std::vector<int> gem1_cand;
    std::vector<int> gem2_cand;

    MatchedIndex()
    : hycal(-1), gem1(-1), gem2(-1)
    {};
    MatchedIndex(int idx)
    : hycal(idx), gem1(-1), gem2(-1)
    {};
};

class PRadDetMatch : public ConfigObject
{
public:
    PRadDetMatch(const std::string &path = "");
    virtual ~PRadDetMatch();

    void Configure(const std::string& path);

    std::vector<MatchedIndex> Match(std::vector<HyCalCluster> &hycal,
                                    std::vector<GEMCluster> &gem1,
                                    std::vector<GEMCluster> &gem2) const;
    bool PreMatch(const HyCalCluster &h, const GEMCluster &g) const;
    bool PostMatch(MatchedIndex &idx, HyCalCluster &h, GEMCluster *g1, GEMCluster *g2) const;

private:
    float gemRes;
    float leadGlassRes;
    float crystalRes;
    float transitionRes;
    float matchSigma;
    float overlapSigma;
};

#endif
