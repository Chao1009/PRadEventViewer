//============================================================================//
// The data handler and container class                                       //
// Dealing with the data from all the channels                                //
//                                                                            //
// Chao Peng                                                                  //
// 02/07/2016                                                                 //
//============================================================================//

#include <iostream>
#include <iomanip>
#include <algorithm>
#include "PRadDataHandler.h"
#include "PRadEvioParser.h"
#include "PRadDSTParser.h"
#include "PRadHyCalSystem.h"
#include "PRadGEMSystem.h"
#include "PRadBenchMark.h"
#include "ConfigParser.h"
#include "TH2.h"

#define EPICS_UNDEFINED_VALUE -9999.9

using namespace std;

PRadDataHandler::PRadDataHandler()
: parser(new PRadEvioParser(this)), dst_parser(new PRadDSTParser(this)),
  hycal_sys(nullptr), gem_sys(nullptr), totalE(0), onlineMode(false),
  replayMode(false), current_event(0)
{
    TagEHist = new TH2I("Tagger E", "Tagger E counter", 2000, 0, 20000, 384, 0, 383);
    TagTHist = new TH2I("Tagger T", "Tagger T counter", 2000, 0, 20000, 128, 0, 127);

    onlineInfo.add_trigger("Lead Glass Sum", 0);
    onlineInfo.add_trigger("Total Sum", 1);
    onlineInfo.add_trigger("LMS Led", 2);
    onlineInfo.add_trigger("LMS Alpha Source", 3);
    onlineInfo.add_trigger("Tagger Master OR", 4);
    onlineInfo.add_trigger("Scintillator", 5);
}

PRadDataHandler::~PRadDataHandler()
{
    delete TagEHist;
    delete TagTHist;

    delete parser;
    delete dst_parser;
}

void PRadDataHandler::ReadConfig(const string &path)
{
    ConfigParser c_parser;
    c_parser.SetSplitters(":,");

    if(!c_parser.ReadFile(path)) {
        cerr << "Data Handler: Cannot open configuration file "
             << "\"" << path << "\"."
             << endl;
    }

    while(c_parser.ParseLine())
    {
        string func_name = c_parser.TakeFirst();
        if((func_name.find("EPICS Channel") != string::npos)) {
            const string var1 = c_parser.TakeFirst().String();
            ExecuteConfigCommand(&PRadDataHandler::ReadEPICSChannels, var1);
        }
        if((func_name.find("Run Number") != string::npos)) {
            const int var1 = c_parser.TakeFirst().Int();
            ExecuteConfigCommand(&PRadDataHandler::SetRunNumber, var1);
        }
        if((func_name.find("Initialize File") != string::npos)) {
            const string var1 = c_parser.TakeFirst().String();
            ExecuteConfigCommand(&PRadDataHandler::InitializeByData, var1, -1, 2);
        }
    }
}

// execute command
template<typename... Args>
void PRadDataHandler::ExecuteConfigCommand(void (PRadDataHandler::*act)(Args...), Args&&... args)
{
    (this->*act)(forward<Args>(args)...);
}

// decode event buffer
void PRadDataHandler::Decode(const void *buffer)
{
    parser->ReadEventBuffer(buffer);

    WaitEventProcess();
}

void PRadDataHandler::SetOnlineMode(const bool &mode)
{
    onlineMode = mode;
}

void PRadDataHandler::RegisterEPICS(const string &name, const uint32_t &id, const float &value)
{
    if(id >= (uint32_t)epics_values.size())
    {
        epics_values.resize(id + 1, EPICS_UNDEFINED_VALUE);
    }

    epics_map[name] = id;
    epics_values.at(id) = value;
}

// erase the data container
void PRadDataHandler::Clear()
{
    // used memory won't be released, but it can be used again for new data file
    energyData = deque<EventData>();
    epicsData = deque<EPICSData>();
    runInfo.clear();

    parser->SetEventNumber(0);
    totalE = 0;

    TagEHist->Reset();
    TagTHist->Reset();

    if(hycal_sys)
        hycal_sys->Reset();

    if(gem_sys)
        gem_sys->Reset();
}

void PRadDataHandler::UpdateTrgType(const unsigned char &trg)
{
    if(newEvent->trigger && (newEvent->trigger != trg)) {
        cerr << "ERROR: Trigger type mismatch at event "
             << parser->GetEventNumber()
             << ", was " << (int) newEvent->trigger
             << " now " << (int) trg
             << endl;
    }
    newEvent->trigger = trg;
}

void PRadDataHandler::UpdateEPICS(const string &name, const float &value)
{
    auto it = epics_map.find(name);

    if(it == epics_map.end()) {
        cout << "Data Handler:: Received data from unregistered EPICS channel " << name
             << ". Assign a new channel id " << epics_values.size()
             << "." << endl;
        epics_map[name] = epics_values.size();
        epics_values.push_back(value);
    } else {
        epics_values.at(it->second) = value;
    }
}

void PRadDataHandler::AccumulateBeamCharge(EventData &event)
{
    if(event.is_physics_event())
        runInfo.beam_charge += event.get_beam_charge();
}

void PRadDataHandler::UpdateLiveTimeScaler(EventData &event)
{
    if(event.is_physics_event()) {
        runInfo.ungated_count += event.get_ref_channel().ungated_count;
        runInfo.dead_count += event.get_ref_channel().gated_count;
    }
}

void PRadDataHandler::UpdateOnlineInfo(EventData &event)
{
    // update triggers
    for(auto trg_ch : onlineInfo.trigger_info)
    {
        if(trg_ch.id < event.dsc_data.size())
        {
            // get ungated trigger counts
            unsigned int counts = event.get_dsc_channel(trg_ch.id).ungated_count;

            // calculate the frequency
            trg_ch.freq = (double)counts / event.get_beam_time();
        }

        else {
            cerr << "Data Handler: Unmatched discriminator data from event "
                 << event.event_number
                 << ", expect trigger " << trg_ch.name
                 << " at channel " << trg_ch.id
                 << ", but the event only has " << event.dsc_data.size()
                 << " dsc channels." << endl;
        }
    }

    // update live time
    onlineInfo.live_time = event.get_live_time();

    //update beam current
    onlineInfo.beam_current = event.get_beam_current();
}

float PRadDataHandler::GetEPICSValue(const string &name)
{
    auto it = epics_map.find(name);
    if(it == epics_map.end()) {
        cerr << "Data Handler: Did not find EPICS channel " << name << endl;
        return EPICS_UNDEFINED_VALUE;
    }

    return epics_values.at(it->second);
}

float PRadDataHandler::GetEPICSValue(const string &name, const int &index)
{
    if((unsigned int)index >= energyData.size())
        return GetEPICSValue(name);

    return GetEPICSValue(name, energyData.at(index));
}

float PRadDataHandler::GetEPICSValue(const string &name, const EventData &event)
{
    float result = EPICS_UNDEFINED_VALUE;

    auto it = epics_map.find(name);
    if(it == epics_map.end()) {
        cerr << "Data Handler: Did not find EPICS channel " << name << endl;
        return result;
    }

    uint32_t channel_id = it->second;
    int event_number = event.event_number;

    // find the epics event before this event
    for(size_t i = 0; i < epicsData.size(); ++i)
    {
        if(epicsData.at(i).event_number >= event_number) {
            if(i > 0) result = epicsData.at(i-1).values.at(channel_id);
            break;
        }
    }

    return result;
}

void PRadDataHandler::FeedData(JLabTIData &tiData)
{
    newEvent->timestamp = tiData.time_high;
    newEvent->timestamp <<= 32;
    newEvent->timestamp |= tiData.time_low;
}

void PRadDataHandler::FeedData(JLabDSCData &dscData)
{
    for(uint32_t i = 0; i < dscData.size; ++i)
    {
        newEvent->dsc_data.emplace_back(dscData.gated_buf[i], dscData.ungated_buf[i]);
    }
}

// feed ADC1881M data
void PRadDataHandler::FeedData(ADC1881MData &adcData)
{
    if(!hycal_sys)
        return;

    // get the channel
    PRadADCChannel *channel = hycal_sys->GetADCChannel(adcData.addr);

    if(!channel)
        return;

    if(newEvent->is_physics_event()) {
        if(channel->Sparsify(adcData.val)) {
            newEvent->add_adc(ADC_Data(channel->GetID(), adcData.val)); // store this data word
        }
    } else if (newEvent->is_monitor_event()) {
        newEvent->add_adc(ADC_Data(channel->GetID(), adcData.val));
    }

}

void PRadDataHandler::FeedData(TDCV767Data &tdcData)
{
    if(!hycal_sys)
        return;

    PRadTDCChannel *tdc = hycal_sys->GetTDCChannel(tdcData.addr);

    if(!tdc)
        return;

    newEvent->tdc_data.push_back(TDC_Data(tdc->GetID(), tdcData.val));
}

void PRadDataHandler::FeedData(TDCV1190Data &tdcData)
{
    if(!hycal_sys)
        return;

    if(tdcData.addr.crate != PRadTS) {
        FeedTaggerHits(tdcData);
        return;
    }

    PRadTDCChannel *tdc = hycal_sys->GetTDCChannel(tdcData.addr);

    if(!tdc)
        return;

    newEvent->add_tdc(TDC_Data(tdc->GetID(), tdcData.val));
}

void PRadDataHandler::FeedTaggerHits(TDCV1190Data &tdcData)
{
#define TAGGER_CHANID 30000 // Tagger tdc id will start from this number
#define TAGGER_T_CHANID 1000 // Start from TAGGER_CHANID, more than 1000 will be t channel
    // E channel
    if(tdcData.addr.slot == 3 || tdcData.addr.slot == 5 || tdcData.addr.slot == 7)
    {
        int e_ch = tdcData.addr.channel + (tdcData.addr.slot - 3)*64 + TAGGER_CHANID;
        // E Channel 30000 + channel
        newEvent->add_tdc(TDC_Data(e_ch, tdcData.val));
    }
    // T Channel
    if(tdcData.addr.slot == 14)
    {
        int t_lr = tdcData.addr.channel/64;
        int t_ch = tdcData.addr.channel%64;
        if(t_ch > 31)
            t_ch = 32 + (t_ch + 16)%32;
        else
            t_ch = (t_ch + 16)%32;
        t_ch += t_lr*64;
        newEvent->add_tdc(TDC_Data(t_ch + TAGGER_CHANID + TAGGER_T_CHANID, tdcData.val));
    }
}

// feed GEM data
void PRadDataHandler::FeedData(GEMRawData &gemData)
{
    if(gem_sys)
        gem_sys->FillRawData(gemData, newEvent->gem_data, newEvent->is_monitor_event());
}

// feed GEM data which has been zero-suppressed
void PRadDataHandler::FeedData(vector<GEMZeroSupData> &gemData)
{
    if(gem_sys)
        gem_sys->FillZeroSupData(gemData, newEvent->gem_data);
}

//TODO move to hycal system
void PRadDataHandler::FillHistograms(EventData &data)
{
    if(!hycal_sys)
        return;

    double energy = 0.;

    // for all types of events
    for(auto &adc : data.adc_data)
    {
        PRadADCChannel *channel = hycal_sys->GetADCChannel(adc.channel_id);
        if(!channel)
            continue;

        channel->FillHist(adc.value, data.trigger);
        energy += channel->GetEnergy(adc.value);
    }

    if(!data.is_physics_event())
        return;

    // for only physics events
    hycal_sys->FillEnergyHist(energy);

    for(auto &tdc : data.tdc_data)
    {
        PRadTDCChannel *channel = hycal_sys->GetTDCChannel(tdc.channel_id);
        if(channel) {
            channel->FillHist(tdc.value);
        } else if(tdc.channel_id >= TAGGER_CHANID) {
            int id = tdc.channel_id - TAGGER_CHANID;
            if(id >= TAGGER_T_CHANID)
                TagTHist->Fill(tdc.value, id - TAGGER_T_CHANID);
            else
                TagEHist->Fill(tdc.value, id - TAGGER_CHANID);
        }
    }
}

// signal of new event
void PRadDataHandler::StartofNewEvent(const unsigned char &tag)
{
    newEvent = new EventData(tag);
}

// signal of event end, save event or discard event in online mode
void PRadDataHandler::EndofThisEvent(const unsigned int &ev)
{
    newEvent->event_number = ev;
    // wait for the process thread
    WaitEventProcess();

    end_thread = thread(&PRadDataHandler::EndProcess, this, newEvent);
}

void PRadDataHandler::WaitEventProcess()
{
    if(end_thread.joinable())
        end_thread.join();
}

void PRadDataHandler::EndProcess(EventData *data)
{
    if(data->type == EPICS_Info) {

        if(onlineMode && epicsData.size())
            epicsData.pop_front();

        if(replayMode)
            dst_parser->WriteEPICS(EPICSData(data->event_number, epics_values));
        else
            epicsData.emplace_back(data->event_number, epics_values);

    } else { // event or sync event

        FillHistograms(*data);

        if(data->type == CODA_Sync) {
            AccumulateBeamCharge(*data);
            UpdateLiveTimeScaler(*data);
            if(onlineMode)
                UpdateOnlineInfo(*data);
        }

        if(onlineMode && energyData.size()) // online mode only saves the last event, to reduce usage of memory
            energyData.pop_front();

        if(replayMode)
            dst_parser->WriteEvent(*data);
        else
            energyData.emplace_back(move(*data)); // save event

    }

    delete data; // new data memory is released here
}

// show the event to event viewer
void PRadDataHandler::ChooseEvent(const int &idx)
{
    if (energyData.size()) { // offline mode, pick the event given by console
        if((unsigned int) idx >= energyData.size())
            ChooseEvent(energyData.back());
        else
            ChooseEvent(energyData.at(idx));
    }
}

void PRadDataHandler::ChooseEvent(const EventData &event)
{
    if(hycal_sys)
        hycal_sys->ChooseEvent(event);
    if(gem_sys)
        gem_sys->ChooseEvent(event);

    current_event = event.event_number;
}

double PRadDataHandler::GetEnergy(const EventData &event)
{
    if(!hycal_sys)
        return 0.;

    double energy = 0.;
    for(auto &adc : event.adc_data)
    {
        PRadADCChannel *channel = hycal_sys->GetADCChannel(adc.channel_id);
        if(channel)
            energy += channel->GetEnergy(adc.value);
    }

    return energy;
}

int PRadDataHandler::GetCurrentEventNb()
{
    return current_event;
}

vector<epics_ch> PRadDataHandler::GetSortedEPICSList()
{
    vector<epics_ch> epics_list;

    for(auto &ch : epics_map)
    {
        float value = epics_values.at(ch.second);
        epics_list.emplace_back(ch.first, ch.second, value);
    }

    sort(epics_list.begin(), epics_list.end(), [](const epics_ch &a, const epics_ch &b) {return a.id < b.id;});

    return epics_list;
}

void PRadDataHandler::PrintOutEPICS()
{
    vector<epics_ch> epics_list = GetSortedEPICSList();

    for(auto &ch : epics_list)
    {
        cout << ch.name << ": " << epics_values.at(ch.id) << endl;
    }
}

void PRadDataHandler::PrintOutEPICS(const string &name)
{
    auto it = epics_map.find(name);
    if(it == epics_map.end()) {
        cout << "Did not find the EPICS channel "
             << name << endl;
        return;
    }

    cout << name << ": " << epics_values.at(it->second) << endl;
}

void PRadDataHandler::SaveEPICSChannels(const string &path)
{
    ofstream out(path);

    if(!out.is_open()) {
        cerr << "Cannot open file "
             << "\"" << path << "\""
             << " to save EPICS channels!"
             << endl;
        return;
    }

    vector<epics_ch> epics_list = GetSortedEPICSList();

    for(auto &ch : epics_list)
    {
        out << ch.name << endl;
    }

    out.close();
}

EventData &PRadDataHandler::GetEvent(const unsigned int &index)
throw (PRadException)
{
    if(!energyData.size())
        throw PRadException("PRad Data Handler Error", "Empty data bank!");

    if(index >= energyData.size()) {
        return energyData.back();
    } else {
        return energyData.at(index);
    }
}

EPICSData &PRadDataHandler::GetEPICSEvent(const unsigned int &index)
{
    if(index >= epicsData.size()) {
        return epicsData.back();
    } else {
        return epicsData.at(index);
    }
}

void PRadDataHandler::ReadEPICSChannels(const string &path)
{
    ConfigParser c_parser;

    if(!c_parser.ReadFile(path)) {
        cout << "WARNING: Fail to open EPICS channel file "
             << "\"" << path << "\""
             << ", no EPICS channel created!"
             << endl;
        return;
    }

    string name;
    float initial_value = EPICS_UNDEFINED_VALUE;

    while(c_parser.ParseLine())
    {
        if(c_parser.NbofElements() == 1) {
            name = c_parser.TakeFirst();
            if(epics_map.find(name) == epics_map.end()) {
                epics_map[name] = epics_values.size();
                epics_values.push_back(initial_value);
            } else {
                cout << "Duplicated epics channel " << name
                     << ", its channel id is " << epics_map[name]
                     << endl;
            }
        } else {
            cout << "Unrecognized input format in  epics channel file, skipped one line!"
                 << endl;
        }
    }

};

// TODO move to hycal system
// Refill energy hist after correct gain factos
void PRadDataHandler::RefillEnergyHist()
{
    if(!hycal_sys)
        return;

    hycal_sys->ResetEnergyHist();

    for(auto &event : energyData)
    {
        if(!event.is_physics_event())
            continue;

        hycal_sys->FillEnergyHist(GetEnergy(event));
    }
}

void PRadDataHandler::ReadFromEvio(const string &path, const int &evt, const bool &verbose)
{
    parser->ReadEvioFile(path.c_str(), evt, verbose);
    WaitEventProcess();
}

void PRadDataHandler::ReadFromSplitEvio(const string &path, const int &split, const bool &verbose)
{
    if(split < 0) {// default input, no split
        ReadFromEvio(path.c_str(), -1, verbose);
    } else {
        for(int i = 0; i <= split; ++i)
        {
            string split_path = path + "." + to_string(i);
            ReadFromEvio(split_path.c_str(), -1, verbose);
        }
    }
}

void PRadDataHandler::InitializeByData(const string &path, int run, int ref)
{
    PRadBenchMark timer;

    if(!hycal_sys || !gem_sys) {
        cout << "Data Handler: HyCal System or GEM System missing, abort initialization."
             << endl;
        return;
    }

    cout << "Data Handler: Initializing from Data File "
         << "\"" << path << "\"."
         << endl;

    if(!path.empty()) {
        // auto update run number
        if(run < 0)
            GetRunNumberFromFileName(path);
        else
            SetRunNumber(run);

        gem_sys->SetPedestalMode(true);

        parser->ReadEvioFile(path.c_str(), 20000);
    }

    cout << "Data Handler: Fitting Pedestal for HyCal." << endl;
    hycal_sys->FitPedestal();

    cout << "Data Handler: Correct HyCal Gain Factor, Run Number: " << runInfo.run_number << "." << endl;
    hycal_sys->CorrectGainFactor(ref);

    cout << "Data Handler: Fitting Pedestal for GEM." << endl;
    gem_sys->FitPedestal();
//    gem_sys->SavePedestal("gem_ped_" + to_string(runInfo.run_number) + ".dat");
//    gem_sys->SaveHistograms("gem_ped_" + to_string(runInfo.run_number) + ".root");

    cout << "Data Handler: Releasing Memeory." << endl;
    gem_sys->SetPedestalMode(false);

    // save run number
    int run_number = runInfo.run_number;
    Clear();
    SetRunNumber(run_number);

    cout << "Data Handler: Done initialization, took " << timer.GetElapsedTime()/1000. << " s" << endl;
}

void PRadDataHandler::GetRunNumberFromFileName(const string &name, const size_t &pos, const bool &verbose)
{
    // get rid of suffix
    auto nameEnd = name.find(".evio");

    if(nameEnd == string::npos)
        nameEnd = name.size();
    else
        nameEnd -= 1;

    // get rid of directories
    auto nameBeg = name.find_last_of("/");
    if(nameBeg == string::npos)
        nameBeg = 0;
    else
        nameBeg += 1;

    int number = ConfigParser::find_integer(name.substr(nameBeg, nameEnd - nameBeg + 1), pos);

    if(number > 0) {

        if(verbose) {
            cout << "Data Handler: Run number is automatcially determined from file name."
                 << endl
                 << "File name: " << name
                 << endl
                 << "Run number: " << number
                 << endl;
        }

        SetRunNumber(number);
    }
}

// find event by its event number
// it is assumed the files decoded are all from 1 single run and they are loaded in order
// otherwise this function will not work properly
int PRadDataHandler::FindEventIndex(const int &ev)
{
    int result = -1;

    if(ev < 0) {
        cout << "Data Handler: Cannot find event with negative event number!" << endl;
        return result;
    }

    if(!energyData.size()) {
        cout << "Data Handler: No event found since data bank is empty." << endl;
        return result;
    }

    int data_begin = 0;
    int data_end = (int)energyData.size() - 1;

    int first_event = energyData.at(0).event_number;

    if(ev > first_event) {
        int index = ev - first_event;
        if(index >= data_end)
            index = data_end;
        int diff = energyData.at(index).event_number - ev;
        while(diff > 0)
        {
            index -= diff;
            if(index <= 0) {
                index = 0;
                break;
            }
            diff = energyData.at(index).event_number - ev;
        }

        data_begin = index;
    }

    for(int i = data_begin; i < data_end; ++i) {
        if(energyData.at(i) == ev)
            return i;
    }

    return result;
}

void PRadDataHandler::Replay(const string &r_path, const int &split, const string &w_path)
{
    if(w_path.empty()) {
        string file = "prad_" + to_string(runInfo.run_number) + ".dst";
        dst_parser->OpenOutput(file);
    } else {
        dst_parser->OpenOutput(w_path);
    }

    cout << "Replay started!" << endl;
    PRadBenchMark timer;

    dst_parser->WriteHyCalInfo(hycal_sys);
    dst_parser->WriteGEMInfo(gem_sys);
    dst_parser->WriteEPICSMap();

    replayMode = true;

    ReadFromSplitEvio(r_path, split);

    dst_parser->WriteRunInfo();

    replayMode = false;

    cout << "Replay done, took " << timer.GetElapsedTime()/1000. << " s!" << endl;
    dst_parser->CloseOutput();
}

void PRadDataHandler::ReadFromDST(const string &path, const uint32_t &mode)
{
    try {
        dst_parser->OpenInput(path);

        dst_parser->SetMode(mode);

        cout << "Data Handler: Reading events from DST file "
             << "\"" << path << "\""
             << endl;

        while(dst_parser->Read())
        {
            switch(dst_parser->EventType())
            {
            case PRad_DST_Event:
                FillHistograms(dst_parser->GetEvent());
                energyData.push_back(dst_parser->GetEvent());
                break;
            case PRad_DST_Epics:
                epicsData.push_back(dst_parser->GetEPICSEvent());
                break;
            default:
                break;
            }
        }

    } catch(PRadException &e) {
        cerr << e.FailureType() << ": "
             << e.FailureDesc() << endl
             << "Write to DST Aborted!" << endl;
    } catch(exception &e) {
        cerr << e.what() << endl
             << "Write to DST Aborted!" << endl;
    }
    dst_parser->CloseInput();
 }

void PRadDataHandler::WriteToDST(const string &path)
{
    try {
        dst_parser->OpenOutput(path);

        cout << "Data Handler: Saving DST file "
             << "\"" << path << "\""
             << endl;

        dst_parser->WriteHyCalInfo(hycal_sys);
        dst_parser->WriteGEMInfo(gem_sys);
        dst_parser->WriteEPICSMap();

        for(auto &epics : epicsData)
        {
            dst_parser->WriteEPICS(epics);
        }

        for(auto &event : energyData)
        {
            dst_parser->WriteEvent(event);
        }

        dst_parser->WriteRunInfo();

    } catch(PRadException &e) {
        cerr << e.FailureType() << ": "
             << e.FailureDesc() << endl
             << "Write to DST Aborted!" << endl;
    } catch(exception &e) {
        cerr << e.what() << endl
             << "Write to DST Aborted!" << endl;
    }

    dst_parser->CloseOutput();
}
