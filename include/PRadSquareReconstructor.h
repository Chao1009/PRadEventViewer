#ifndef PRAD_RECONSTRUCTOR_H
#define PRAD_RECONSTRUCTOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include "PRadEventStruct.h"
#include "ConfigParser.h"

class PRadDataHandler;
class PRadDAQUnit;
class PRadIslandWrapper;

class PRadReconstructor
{
public:
    PRadReconstructor(PRadDataHandler *h = nullptr);
    virtual ~PRadReconstructor();

    void InitConfig(const std::string &path);
    ConfigValue GetConfigValue(const std::string &var_name);
    void Clear();
    void SetHandler(PRadDataHandler *h);
    std::vector<HyCalHit> &CoarseHyCalReconstruct(const int &event_index = -1);
    std::vector<HyCalHit> &CoarseHyCalReconstruct(EventData &event);
    std::vector<HyCalHit> &IslandReconstruct(const int &event_index = -1);
    std::vector<HyCalHit> &IslandReconstruct(EventData &event);
    unsigned short GetHighestModuleID() { return fHighestModuleID; }
    unsigned short FindClusterCenterModule (double x, double y);
    std::vector<HyCalHit> &Reconstruct_fivebyfive();

protected:
    unsigned short getMaxEChannel();
    //void GEMCoorToLab(float* x, float *y, int type);
    //void HyCalCoorToLab(float* x, float *y);
    bool useLogWeight(double x, double y);
    std::vector<unsigned short> findCluster(unsigned short cneterID, double &clusterEnergy);
    std::vector<unsigned short> &GetTimeForCluster(unsigned short channelID);

    PRadDataHandler *fHandler;
    PRadIslandWrapper *fIsland;
    std::vector<unsigned short> fClusterCenterID;
    std::vector<HyCalHit> fHyCalHit;
    std::vector<float>* fGEM1XHit;
    std::vector<float>* fGEM1YHit;
    std::vector<float>* fGEM2XHit;
    std::vector<float>* fGEM2YHit;
    unsigned short fHighestModuleID;

    //for parameter from reconstruction data base
    int fMaxNCluster;
    double fMinClusterCenterE;
    double fMinClusterE;

    //some universal constants
    double fMoliereCrystal;
    double fMoliereLeadGlass;
    double fMoliereRatio;
    double fBaseR;
    double fCorrectFactor;

    //configuration map
    std::unordered_map<std::string, ConfigValue> fConfigMap;

public:
    static double Distance(PRadDAQUnit *u1, PRadDAQUnit *u2);
    static double Distance(const double &x1, const double &y1, const double &x2, const double &y2);
    static double Distance(const double &x1, const double &y1, const double &x2, const double &y2, const double &z1, const double &z2);
    static double Distance(const std::vector<double> &p1, const std::vector<double> &p2);
};

#endif