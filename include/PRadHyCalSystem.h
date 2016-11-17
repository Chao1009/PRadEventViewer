#ifndef PRAD_HYCAL_SYSTEM_H
#define PRAD_HYCAL_SYSTEM_H

#include <string>
#include <ostream>
#include "PRadEventStruct.h"
#include "PRadHyCalDetector.h"
#include "PRadHyCalCluster.h"
#include "PRadSquareCluster.h"
#include "PRadTDCChannel.h"
#include "PRadADCChannel.h"
#include "ConfigObject.h"

// adc searching speed is important, thus reserve buckets to have unordered_map
// better formed
#define ADC_BUCKETS 5000
// a simple hash function for DAQ configuration
namespace std
{
    template<>
    struct hash<ChannelAddress>
    {
        unsigned int operator()(const ChannelAddress &addr)
        const
        {
            // crate id is 1-6, slot is 1-26, channel is 0-63
            // thus they can be filled in a 14 bit word
            // [ 0 0 0 | 0 0 0 0 0 | 0 0 0 0 0 0 ]
            // [ crate |    slot   |   channel   ]
            // this simple hash ensures no collision for current setup
            return ((addr.crate << 11) | (addr.slot << 6) | addr.channel);
        }
    };
}

class TH1D;

class PRadHyCalSystem : public ConfigObject
{
public:
    // constructor
    PRadHyCalSystem(const std::string &path = "");

    // copy/move constructors
    PRadHyCalSystem(const PRadHyCalSystem &that);
    PRadHyCalSystem(PRadHyCalSystem &&that);

    // destructor
    virtual ~PRadHyCalSystem();

    // copy/move assignment operators
    PRadHyCalSystem &operator =(const PRadHyCalSystem &rhs);
    PRadHyCalSystem &operator =(PRadHyCalSystem &&rhs);

    // configuration
    void Configure(const std::string &path);
    void ReadChannelList(const std::string &path);
    void ReadPedestalFile(const std::string &path);
    void ReadRunInfoFile(const std::string &path);

    // connections
    void BuildConnections();

    // events related
    void ChooseEvent(const EventData &data);
    void Reconstruct();
    void Reconstruct(const EventData &data);
    void Reset();

    // detector related
    void SetDetector(PRadHyCalDetector *h);
    void RemoveDetector();
    void DisconnectDetector(bool force_disconn = false);
    PRadHyCalModule *GetModule(const int &id) const;
    PRadHyCalModule *GetModule(const std::string &name) const;
    std::vector<PRadHyCalModule*> GetModuleList() const;
    PRadHyCalDetector *GetDetector() const {return hycal;};

    // daq related
    bool AddADCChannel(PRadADCChannel *adc);
    bool AddTDCChannel(PRadTDCChannel *tdc);
    void ClearADCChannel();
    void ClearTDCChannel();
    PRadADCChannel *GetADCChannel(const int &id) const;
    PRadADCChannel *GetADCChannel(const std::string &name) const;
    PRadADCChannel *GetADCChannel(const ChannelAddress &addr) const;
    PRadTDCChannel *GetTDCChannel(const int &id) const;
    PRadTDCChannel *GetTDCChannel(const std::string &name) const;
    PRadTDCChannel *GetTDCChannel(const ChannelAddress &addr) const;
    const std::vector<PRadADCChannel*> &GetADCList() const {return adc_list;};
    const std::vector<PRadTDCChannel*> &GetTDCList() const {return tdc_list;};

    // clustering method related
    bool AddClusterMethod(const std::string &name, PRadHyCalCluster *c);
    void RemoveClusterMethod(const std::string &name);
    void ClearClusterMethods();
    void SetClusterMethod(const std::string &name);
    PRadHyCalCluster *GetClusterMethod(const std::string &name) const;
    std::string GetClusterMethodName() const;
    std::vector<std::string> GetClusterMethodNames() const;

    // histogram related
    void FillEnergyHist();
    void FillEnergyHist(const double &e);
    void ResetEnergyHist();
    TH1 *GetEnergyHist() const {return energy_hist;};
    void SaveHists(const std::string &path);
    std::vector<double> FitHist(const std::string &channel,
                                const std::string &hist_name,
                                const std::string &fit_function,
                                const double &range_min,
                                const double &range_max,
                                const bool &verbose) const throw(PRadException);
    void FitPedestal();
    void CorrectGainFactor(int ref);


private:
    PRadHyCalDetector *hycal;
    PRadHyCalCluster *recon;
    TH1D *energy_hist;

    // channel lists
    std::vector<PRadADCChannel*> adc_list;
    std::vector<PRadTDCChannel*> tdc_list;

    // channel maps
    std::unordered_map<ChannelAddress, PRadADCChannel*> adc_addr_map;
    std::unordered_map<std::string, PRadADCChannel*> adc_name_map;
    std::unordered_map<ChannelAddress, PRadTDCChannel*> tdc_addr_map;
    std::unordered_map<std::string, PRadTDCChannel*> tdc_name_map;

    // clustering method map
    std::unordered_map<std::string, PRadHyCalCluster*> recon_map;
};

#endif
