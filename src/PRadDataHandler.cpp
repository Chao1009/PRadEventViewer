//============================================================================//
// The data handler and container class                                       //
// Dealing with the data from all the channels                                //
// Provided multi-thread support, can be disabled by comment the definition   //
// in PRadDataHandler.h                                                       //
//                                                                            //
// Chao Peng                                                                  //
// 02/07/2016                                                                 //
//============================================================================//

#include <iostream>
#include "PRadDataHandler.h"
#include "PRadEvioParser.h"
#include "HyCalModule.h"
#include "PRadTDCGroup.h"
#include "TH1.h"

#define recon_test

#ifdef recon_test
#include <fstream>
#include "HyCalClusters.h"
#endif

PRadDataHandler::PRadDataHandler()
: totalE(0), onlineMode(false)
{
    // total energy histogram
    energyHist = new TH1D("HyCal Energy", "Total Energy (MeV)", 2500, 0, 2500);
}

PRadDataHandler::~PRadDataHandler()
{
    delete energyHist;

    for(auto &ele : freeList)
    {
        delete ele, ele = nullptr;
    }

    for(auto &it : map_name_tdc)
    {
        delete it.second, it.second = nullptr;
    }
}

// add DAQ channels
void PRadDataHandler::AddChannel(PRadDAQUnit *channel)
{
    RegisterChannel(channel);
    freeList.push_back(channel);
}

// register DAQ channels, memory is managed by other process
void PRadDataHandler::RegisterChannel(PRadDAQUnit *channel)
{
    channel->AssignID(channelList.size());
    channelList.push_back(channel);
}

void PRadDataHandler::AddTDCGroup(PRadTDCGroup *group)
{
    map_name_tdc[group->GetName()] = group;
    map_daq_tdc[group->GetAddress()] = group;
}

void PRadDataHandler::BuildChannelMap()
{
    // build unordered maps separately improves its access speed
    // name map
    for(auto &channel : channelList)
        map_name[channel->GetName()] = channel;

    // DAQ configuration map
    for(auto &channel : channelList)
        map_daq[channel->GetDAQInfo()] = channel;

    // TDC groups
    for(auto &channel : channelList)
    {
        string tdcName = channel->GetTDCName();
        if(tdcName.empty() || tdcName == "N/A" || tdcName == "NONE")
            continue; // not belongs to any tdc group
        PRadTDCGroup *tdcGroup = GetTDCGroup(tdcName);
        if(tdcGroup == nullptr) {
            cerr << "Cannot find TDC group: " << tdcName
                 << " make sure you added all the tdc groups"
                 << endl;
            continue;
        }
        tdcGroup->AddChannel(channel);
    }
}

// erase the data container
void PRadDataHandler::Clear()
{
    // used memory won't be released, but it can be used again for new data file
    energyData.erase(energyData.begin(), energyData.end());
    totalE = 0;
    newEvent.clear();
    energyHist->Reset();
}

void PRadDataHandler::FeedData(JLabTIData &/*tiData*/)
{
    // place holder
}

// feed ADC1881M data
void PRadDataHandler::FeedData(ADC1881MData &adcData)
{
    // find the channel with this DAQ configuration
    daq_iter it = map_daq.find(adcData.config);

    // did not find any channel
    if(it == map_daq.end())
        return;

    // get the channel
    PRadDAQUnit *channel = it->second;

    // fill adc value histogram
    channel->GetHist()->Fill(adcData.val);

    // zero suppression
    unsigned short sparVal = channel->Sparsification(adcData.val);

    if(sparVal) // only store events above pedestal in memory
    {
        ChannelData word(channel->GetID(), sparVal); // save id because it saves memory
#ifdef MULTI_THREAD
        // unfortunately, we have some non-local variable to deal with
        // so lock the thread to prevent concurrent access
        myLock.lock();
#endif
        totalE += channel->Calibration(sparVal); // calculate total energy of this event
        newEvent.push_back(word); // store this data word
#ifdef MULTI_THREAD
        myLock.unlock();
#endif
    }

}

// feed GEM data
void PRadDataHandler::FeedData(GEMAPVData & /*gemData*/)
{
    // implement later
}

void PRadDataHandler::FeedData(TDCV767Data &tdcData)
{
    tdc_daq_iter it = map_daq_tdc.find(tdcData.config);
    if(it == map_daq_tdc.end())
        return;

    PRadTDCGroup *tdc = it->second;
    tdc->GetHist()->Fill(tdcData.val);
}

// update High Voltage
void PRadDataHandler::FeedData(CAENHVData &hvData)
{
    name_iter it = map_name.find(hvData.name);

    if(it == map_name.end())
        return;

    PRadDAQUnit *channel = it->second;
    HyCalModule *module = dynamic_cast<HyCalModule *>(channel);
    if(module == nullptr)
        return;

    if(module->GetHVInfo() == hvData.config) {
        module->UpdateHV(hvData.Vmon, hvData.Vset, hvData.ON);
    } else {
        cerr << "ERROR: incorrect HV Configuration! "
             << "Module: " << hvData.name << endl;
        return;
    }

}

// signal of event end, save event or discard event in online mode
void PRadDataHandler::EndofThisEvent()
{
    if(onlineMode) { // online mode only saves the last event, to reduce usage of memory
        lastEvent = newEvent;
    } else {
        energyData.push_back(newEvent); // save event
    }

    energyHist->Fill(totalE); // fill energy histogram

#ifdef recon_test
    ofstream outfile;
    outfile.open("HyCal_Hits.txt", ofstream::app);
    // reconstruct it
    HyCalClusters cluster;
    for(auto &channel : newEvent.channels)
    {
        HyCalModule *module = dynamic_cast<HyCalModule*>(FindChannel(channel.id));
        if(module)
            cluster.AddModule(module->GetGeometry().x, module->GetGeometry().y, module->Calibration(channel.adcValue));
    }
    vector<HyCalClusters::HyCal_Hits> hits = cluster.ReconstructHits();
    for(auto &hit : hits)
    {
        outfile << energyData.size() << "  " <<  hit.x << "  " << hit.y << "  "  << hit.E << endl;
    }
    outfile.close();
#endif

    // clear buffer for next event
    newEvent.clear();
    totalE = 0;
}

// show the event to event viewer
void PRadDataHandler::UpdateEvent(int idx)
{

    EventData event;

    // != avoids operator definition for non-standard map
    for(auto &channel : channelList)
    {
        channel->UpdateEnergy(0);
    }

    if(onlineMode) { // online mode only show the last event
        event = lastEvent;
    } else { // offline mode, pick the event given by console
        if((unsigned int)idx >= energyData.size())
            return;
        event = energyData[idx];
    }

    for(auto &channel : event.channels)
    {
        channelList[channel.id]->UpdateEnergy(channel.adcValue);
    }

}

// find channels
PRadDAQUnit *PRadDataHandler::FindChannel(const ChannelAddress &daqInfo)
{
    daq_iter it = map_daq.find(daqInfo);
    if(it == map_daq.end())
        return nullptr;
    return it->second;
}

PRadDAQUnit *PRadDataHandler::FindChannel(const string &name)
{
    name_iter it = map_name.find(name);
    if(it == map_name.end())
        return nullptr;
    return it->second;
}

PRadDAQUnit *PRadDataHandler::FindChannel(const unsigned short &id)
{
    if(id >= channelList.size())
        return nullptr;
    return channelList[id];
}

PRadTDCGroup *PRadDataHandler::GetTDCGroup(const string &name)
{
    tdc_name_iter it = map_name_tdc.find(name);
    if(it == map_name_tdc.end())
        return nullptr; // return empty vector
    return it->second;
}

PRadTDCGroup *PRadDataHandler::GetTDCGroup(const ChannelAddress &addr)
{
    tdc_daq_iter it = map_daq_tdc.find(addr);
    if(it == map_daq_tdc.end())
        return nullptr;
    return it->second;
}
