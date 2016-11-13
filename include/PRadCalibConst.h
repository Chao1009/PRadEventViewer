#ifndef PRAD_CALIB_CONST_H
#define PRAD_CALIB_CONST_H

#include <vector>
#include "ConfigParser.h"

// LMS1 LMS2 LMS3
#define DEFAULT_REF_NUM 3

class PRadCalibConst
{
public:
    friend class PRadHyCalModule;

public:
    PRadCalibConst(int ref_num = DEFAULT_REF_NUM);
    PRadCalibConst(double f, double e, double nl, const std::vector<double> &g);
    PRadCalibConst(double f, double e, double nl, double *g, int num);

    virtual ~PRadCalibConst();

    void SetCalibConst(double f) {base_factor = f; factor = f;};
    void SetRefGain(double gain, int ref);
    void ClearRefGains();
    void SetCalibEnergy(double energy) {base_energy = energy;};
    void SetNonLinearFactor(double nl) {non_linear = nl;};
    void GainCorrection(double gain, int ref);

    double GetCalibConst() const {return factor;};
    double GetRefGain(int ref) const;
    double GetCalibEnergy() const {return base_energy;};
    double GetNonLinearFactor() const {return non_linear;};
    double Calibration(const double &adc_value) const;

private:
    double factor;
    double base_factor;
    double base_energy;
    std::vector<double> base_gains;
    double non_linear;
};

ConfigParser &operator >>(ConfigParser &p, PRadCalibConst &c);

#endif
