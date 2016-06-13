//============================================================================//
// PRad Cluster Reconstruction                                                //
//                                                                            //
// Weizhi Xiong                                                               //
// 06/10/2016                                                                 //
//============================================================================//

#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include "PRadReconstructor.h"
#include "PRadDataHandler.h"
#include "PRadDAQUnit.h"

using namespace std;

//____________________________________________________________
PRadReconstructor::PRadReconstructor(){
    fMoliereCrystal = 20.5;
    fMoliereLeadGlass = 38.2;
    fMoliereRatio = fMoliereLeadGlass/fMoliereCrystal;
    fBaseR = 60.;
    fCorrectFactor = 1.;
    fMaxNCluster = 10;
    fMinClusterCenterE = 20.;
    fMinClusterE = 100.;
}
//____________________________________________________________
void PRadReconstructor::Clear()
{
    fHyCalHit.clear();
    fClusterCenterID.clear();
    fConfigMap.clear();
}
//____________________________________________________________
void PRadReconstructor::InitConfig(const string &path)
{
    ConfigParser c_parser(": ,\t"); // self-defined splitters

    if(!c_parser.OpenFile(path)) {
         cout << "Cannot open file " << path << endl;
     }
 
    unordered_map<string, float> variable_map;

    while(c_parser.ParseLine())
    {
        if (c_parser.NbofElements() != 2)
            continue;

        string var_name = c_parser.TakeFirst();
        ConfigValue var_value = c_parser.TakeFirst();
        fConfigMap[var_name] = var_value;
    }

    fMaxNCluster = GetConfigValue("MAX_N_CLUSTER").Int();
    fMinClusterCenterE = GetConfigValue("MIN_CLUSTER_CENTER_E").Double();
    fMinClusterE = GetConfigValue("MIN_CLUSTER_E").Double();
}

ConfigValue PRadReconstructor::GetConfigValue(const string &name)
{
    auto it = fConfigMap.find(name);
    if(it == fConfigMap.end())
        return ConfigValue();
    return it->second;
}
//________________________________________________________________
vector<HyCalHit> &PRadReconstructor::CoarseHyCalReconstruct()
{
    Clear(); // clear all the saved buffer before analyzing the next event

    double weightX = 0.;
    double weightY = 0.;
    double totalWeight = 0.;

    for (unsigned short i = 0; i < fMaxNCluster; ++i)
    {
        int theMaxModuleID = getMaxEChannel();
   
        if (theMaxModuleID == 0xffff)
            break;//this happens if no module has large enough energy
    
        double clusterEnergy = 0.;
        vector<unsigned short> collection = findCluster(theMaxModuleID, clusterEnergy);
    
        if (clusterEnergy <= fMinClusterE || collection.size() <= 1) {
            --i;
            continue;
        }

        weightX = 0.;
        weightY = 0.;
        totalWeight = 0.;

        for (unsigned short j=0; j<collection.size(); ++j)
        {
            unsigned short thisID = collection.at(j);
            PRadDAQUnit* thisModule = fModuleList.at(thisID);
            if (!thisModule->IsHyCalModule())
                continue;

            double thisX = thisModule->GetX();
            double thisY = thisModule->GetY();

            double weight = thisModule->GetEnergy()/clusterEnergy;
            if (useLogWeight(thisX, thisY))
                weight = 10. + log(weight);

            if (weight > 0.) {
                weightX += weight*thisX;
                weightY += weight*thisY;
                totalWeight += weight;
            }
        }

        double hitX = weightX/totalWeight;
        double hitY = weightY/totalWeight;
    
        fHyCalHit.push_back(HyCalHit(hitX, hitY, clusterEnergy));
    }
 
    return fHyCalHit; 
}
//____________________________________________________________
unsigned short PRadReconstructor::getMaxEChannel()
{
    double theMaxValue = 0;
    unsigned short theMaxChannelID = 0xffff;
    bool foundNewCenter = false; 
    for (unsigned int i = 0; i < fModuleList.size(); ++i)
    {
    // if have not found the module with maxmimum energy then find it
    // otherwise check if the next maximum is too close to the existing center
        PRadDAQUnit* thisModule = fModuleList.at(i);
        if (!thisModule->IsHyCalModule())
            continue;
        if (fClusterCenterID.empty()) { 
            if ( thisModule->GetEnergy() > theMaxValue && 
                 thisModule->GetEnergy() > fMinClusterCenterE)
            {
                foundNewCenter = true;
                theMaxValue = thisModule->GetEnergy();
                theMaxChannelID = i;
            }
        } else {
            if ( thisModule->GetEnergy() > fMinClusterCenterE) {
                double theClusterRadius = fBaseR;
          
                if ( thisModule->GetType() == PRadDAQUnit::LeadGlass )
                    theClusterRadius = fMoliereRatio*fBaseR;
        
                double distance = 120.;
                bool ok = true;

                for (unsigned int j = 0; j < fClusterCenterID.size(); ++j)
                {
                    PRadDAQUnit* lastCenterModule = fModuleList.at( fClusterCenterID.at(j) );
                    distance = fmin( distance, Distance( thisModule, lastCenterModule ) );
                    if (distance < 2*theClusterRadius) {
                        ok = false;
                       //theMaxChannelID = i;
                    }
                }

                if (ok) {
                    foundNewCenter = true;
                    theMaxChannelID = i;
                }
            }         
        }
    }

    if (foundNewCenter)
        fClusterCenterID.push_back(theMaxChannelID);

    return theMaxChannelID;
}
//__________________________________________________________________________________________
inline bool PRadReconstructor::useLogWeight(double /*x*/, double /*y*/)
{
    return true; //for now
}
//___________________________________________________________________________________________
vector<unsigned short> PRadReconstructor::findCluster(unsigned short centerID, double &clusterEnergy)
{
    double clusterRadius = 0.;
    double centerX = fModuleList.at(centerID)->GetX();
    double centerY = fModuleList.at(centerID)->GetY();
    vector<unsigned short> collection;

    for (unsigned int i = 0; i < fModuleList.size(); ++i)
    {
        PRadDAQUnit* thisModule = fModuleList.at(i);
        if (!thisModule->IsHyCalModule())
            continue;
        if (thisModule->GetType() == 1) {
            clusterRadius = fBaseR;
        } else {
            clusterRadius = fBaseR*fMoliereRatio;
        }

        if ( thisModule->GetEnergy() > 0. && 
             Distance( thisModule->GetX(), thisModule->GetY(), centerX, centerY ) <= clusterRadius )
        {
            clusterEnergy += thisModule->GetEnergy();
            collection.push_back(i);
        }
    }

    return collection;
}
//___________________________________________________________________________________________
void PRadReconstructor::SetHandler(PRadDataHandler *theHandler)
{
    fHandler = theHandler;
    fModuleList = fHandler->GetChannelList();
}
//___________________________________________________________________________________________
double PRadReconstructor::Distance(PRadDAQUnit *u1, PRadDAQUnit *u2)
{
    double x_dis = u1->GetX() - u2->GetX();
    double y_dis = u1->GetY() - u2->GetY();
    return sqrt( x_dis*x_dis + y_dis*y_dis);
}
//___________________________________________________________________________________________
double PRadReconstructor::Distance(const double &x1, const double &y1, const double &x2, const double &y2)
{
    return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}
//___________________________________________________________________________________________
double PRadReconstructor::Distance(const double &x1, const double &y1, const double &x2, const double &y2, const double &z1, const double &z2)
{
    return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) );
}
//___________________________________________________________________________________________
double PRadReconstructor::Distance(const vector<double> &p1, const vector<double> &p2)
{
    if(p1.size() != p2.size()) {
        cerr << "Dimension is different, failed to calculate distance between two points!" << endl;
        return 0.;
    }

    double quadratic_sum = 0.;
    for(size_t i = 0; i < p1.size(); ++i)
    {
        double quadratic = p1.at(i) - p2.at(i);
        quadratic *= quadratic;
        quadratic_sum += quadratic;
    }

    return sqrt(quadratic_sum);
}
