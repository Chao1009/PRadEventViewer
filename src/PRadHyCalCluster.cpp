//============================================================================//
// Basic PRad Cluster Reconstruction Class For HyCal                          //
// Different reconstruction methods can be implemented accordingly            //
//                                                                            //
// Chao Peng, Weizhi Xiong                                                    //
// 09/28/2016                                                                 //
//============================================================================//

#include <cmath>
#include <iostream>
#include <iomanip>
#include "PRadHyCalCluster.h"

PRadHyCalCluster::PRadHyCalCluster(PRadDataHandler *h)
: fHandler(h), fNHyCalClusters(0)
{
    // place holder
}

PRadHyCalCluster::~PRadHyCalCluster()
{
    // place holder
}

void PRadHyCalCluster::Clear()
{
    fNHyCalClusters = 0;
}

void PRadHyCalCluster::SetHandler(PRadDataHandler *h)
{
    fHandler = h;
}

void PRadHyCalCluster::readConfigFile(const std::string &path)
{
    ConfigParser c_parser(": ,\t="); // self-defined splitters

    if(!c_parser.OpenFile(path)) {
        std::cerr << "PRad HyCal Cluster Error: Cannot open file "
                  << "\"" << path << "\""
                  << std::endl;
        return;
    }

    // save the path
    config_path = path;

    // clear the map
    fConfigMap.clear();

    while(c_parser.ParseLine())
    {
        if (c_parser.NbofElements() != 2)
            continue;

        std::string var_name;
        ConfigValue var_value;
        c_parser >> var_name >> var_value;
        fConfigMap[var_name] = var_value;
    }
}

ConfigValue PRadHyCalCluster::getConfigValue(const std::string &name,
                                             const std::string &def_value,
                                             bool verbose)
{
    auto it = fConfigMap.find(name);
    if(it == fConfigMap.end())
    {
        if(verbose)
            std::cout << name
                      << " not defined in configuration file, set to default value "
                      << def_value
                      << std::endl;
        return ConfigValue(def_value);
    }
    return it->second;
}

ConfigValue PRadHyCalCluster::GetConfigValue(const std::string &name)
{
    auto it = fConfigMap.find(name);
    if(it == fConfigMap.end())
        return ConfigValue("0");
    else
        return it->second;
}

void PRadHyCalCluster::SetConfigValue(const std::string &var_name, const ConfigValue &c_value)
{
    fConfigMap[var_name] = c_value;
}

void PRadHyCalCluster::Configure(const std::string & /*path*/)
{
    // to be implemented by methods
}

void PRadHyCalCluster::Reconstruct(EventData & /*event*/)
{
    // to be implemented by methods
}
