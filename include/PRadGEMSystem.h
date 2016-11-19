#ifndef PRAD_GEM_SYSTEM_H
#define PRAD_GEM_SYSTEM_H

#include <string>
#include <list>
#include <vector>
#include <unordered_map>
#include <fstream>
#include "PRadEventStruct.h"
#include "PRadException.h"
#include "PRadGEMDetector.h"
#include "PRadGEMFEC.h"
#include "PRadGEMCluster.h"
#include "ConfigObject.h"

#ifdef MULTI_THREAD
#include <thread>
#include <mutex>
#endif

// fec id should be consecutive from 0
// enlarge this value if there are more FECs
#define MAX_FEC_ID 128

class PRadGEMSystem : public ConfigObject
{
public:
    // constructor
    PRadGEMSystem(const std::string &config_file = "",
                  int daq_cap = MAX_FEC_ID,
                  int det_cap = PRadDetector::Max_Dets);

    // copy/move constructors
    PRadGEMSystem(const PRadGEMSystem &that);
    PRadGEMSystem(PRadGEMSystem &&that);

    // destructor
    virtual ~PRadGEMSystem();

    // copy/move assignment operators
    PRadGEMSystem &operator =(const PRadGEMSystem &rhs);
    PRadGEMSystem &operator =(PRadGEMSystem &&rhs);

    // public member functions
    void RemoveDetector(int det_id);
    void DisconnectDetector(int det_id, bool force_disconn = false);
    void RemoveFEC(int fec_id);
    void DisconnectFEC(int fec_id, bool force_disconn = false);
    void Configure(const std::string &path);
    void ReadMapFile(const std::string &path) throw(PRadException);
    void ReadPedestalFile(const std::string &path) throw(PRadException);
    void Clear();
    void ChooseEvent(const EventData &data);
    void Reconstruct();
    void Reconstruct(const EventData &data);
    void RebuildDetectorMap();
    void RebuildDAQMap();
    void FillRawData(const GEMRawData &raw, EventData &event);
    void FillZeroSupData(const std::vector<GEMZeroSupData> &data_pack, EventData &event);
    void FillZeroSupData(const GEMZeroSupData &data);
    bool Register(PRadGEMDetector *det);
    bool Register(PRadGEMFEC *fec);

    void SetUnivCommonModeThresLevel(const float &thres);
    void SetUnivZeroSupThresLevel(const float &thres);
    void SetUnivTimeSample(const size_t &thres);
    void SetPedestalMode(const bool &m);
    void FitPedestal();
    void Reset();
    void SavePedestal(const std::string &path) const;
    void SaveHistograms(const std::string &path) const;

    PRadGEMCluster *GetClusterMethod() const {return gem_recon;};
    PRadGEMDetector *GetDetector(const int &id) const;
    PRadGEMDetector *GetDetector(const std::string &name) const;
    PRadGEMFEC *GetFEC(const int &id) const;
    PRadGEMAPV *GetAPV(const GEMChannelAddress &addr) const;
    PRadGEMAPV *GetAPV(const int &fec, const int &adc) const;

    std::vector<GEM_Data> GetZeroSupData() const;
    std::vector<PRadGEMAPV*> GetAPVList() const;
    std::vector<PRadGEMFEC*> GetFECList() const {return fec_list;};
    const std::vector<PRadGEMDetector*> &GetDetectorList() const {return det_list;};

private:
    // private member functions
    void buildDetector(std::list<ConfigValue> &det_args);
    void buildPlane(std::list<ConfigValue> &pln_args);
    void buildFEC(std::list<ConfigValue> &fec_args);
    void buildAPV(std::list<ConfigValue> &apv_args);

private:
    PRadGEMCluster *gem_recon;
    bool PedestalMode;
    std::vector<PRadGEMDetector*> det_list;
    std::vector<PRadGEMFEC*> fec_list;

    // maps
    std::vector<PRadGEMFEC*> daq_slots;
    std::vector<PRadGEMDetector*> det_slots;
    std::unordered_map<std::string, PRadGEMDetector*> det_name_map;

    unsigned int def_ts;
    float def_cth;
    float def_zth;

#ifdef MULTI_THREAD
    std::mutex locker;
#endif
};

#endif
