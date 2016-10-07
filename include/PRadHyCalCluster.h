#ifndef PRAD_HYCAL_CLUSTER_H
#define PRAD_HYCAL_CLUSTER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "PRadEventStruct.h"
#include "PRadDataHandler.h"
#include "PRadDAQUnit.h"
#include "PRadTDCGroup.h"
#include "ConfigParser.h"

#define MAX_HCLUSTERS 250 // Maximum storage

class PRadDataHandler;

class PRadHyCalCluster
{
public:
    PRadHyCalCluster(PRadDataHandler *h = nullptr);
    virtual ~PRadHyCalCluster();

    void ReadConfigFile(const std::string &path);
    ConfigValue GetConfigValue(const std::string &var_name,
                               const std::string &def_value,
                               bool verbose = true);
    void SetHandler(PRadDataHandler *h);

    // functions that to be overloaded
    virtual void Configure(const std::string &path);
    virtual void Clear();
    virtual void Reconstruct(EventData &event);
    virtual int GetNClusters() {return fNHyCalClusters;};
    virtual HyCalHit *GetCluster() {return fHyCalCluster;};

protected:
    PRadDataHandler *fHandler;
    // configuration map
    std::unordered_map<std::string, ConfigValue> fConfigMap;
    // result array
    HyCalHit fHyCalCluster[MAX_HCLUSTERS];
    int fNHyCalClusters;
};

#endif
