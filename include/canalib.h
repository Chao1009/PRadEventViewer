#ifndef C_ANA_LIB_H
#define C_ANA_LIB_H
// some useful tools for the whole library

#include <iterator>
#include <algorithm>
#include <utility>
#include <cmath>

// definition of some constants
#define ALPHA 7.297352568E-3    // 1./137.03599911
#define PI 3.1415926535897932   // pi
#define RADDEG 57.2957795131    // rad to degree
#define ELECM 0.510998918       // MeV
#define HBARC 197.326968        // hbar*c (MeV*fm)
#define AMUMEV 931.494043       // MeV per amu

namespace cana
{
    double sigmoid(const double &a, const double &p);
    double gamma(const double &z);
    double spence(const double &z, const double &res = 1e-15);
    double spence_tr(const double &z, const double &res, const int &nmax);
    // simpson integration
    double simpson(double begin, double end, double (*f)(const double&), double step, int Nmin);
    template<class T>
    double simpson(double begin, double end,
                   double (T::*f)(const double&), T *t, double step, int Nmin)
    {
        int Nsteps = (end - begin)/step;
        int Nbins = std::max(Nmin, Nsteps)/2;
        double s = (end - begin)/(double)(2.*Nbins);

        // first bin
        double result = (t->*f)(begin) + 4.*(t->*f)(begin + s) + (t->*f)(end);
        double x = begin + 2.*s;
        int i = 1;
        while(i++ < Nbins)
        {
            result += 2.*(t->*f)(x) + 4.*(t->*f)(x + s);
            x += 2.*s;
        }

        return result*s/3.;
    }

    template<class T, typename... Args>
    double simpson(double begin, double end, double step, int Nmin,
                   double (T::*f)(const double&, const Args& ...), T *t, const Args&... args)
    {
        int Nsteps = (end - begin)/step;
        int Nbins = std::max(Nmin, Nsteps)/2;
        double s = (end - begin)/(double)(2.*Nbins);

        // first bin
        double result =  (t->*f)(begin, args...)
                       + 4.*(t->*f)(begin + s, args...)
                       + (t->*f)(end, args...);
        double x = begin + 2.*s;
        int i = 1;
        while(i++ < Nbins)
        {
            result += 2.*(t->*f)(x, args...) + 4.*(t->*f)(x + s, args...);
            x += 2.*s;
        }

        return result*s/3.;
    }

    // the function is based on c++ source code
    // it adds permutation parity track
    template<class BidirIt>
    bool permutate(BidirIt first, BidirIt last, int &parity)
    {
        if (first == last) return false;
        BidirIt i = last;
        if (first == --i) return false;

        while (true) {
            BidirIt i1, i2;

            i1 = i;
            if (*--i < *i1) {
                i2 = last;
                while (!(*i < *--i2))
                    ;
                std::iter_swap(i, i2);
                std::reverse(i1, last);
                size_t swap = std::distance(i1, last)/2 + 1;
                // odd number of swaps
                if(swap&1)  parity *= -1;
                // even number of swaps, no change needed
                return true;
            }
            if (i == first) {
                std::reverse(first, last);
                size_t swap = std::distance(first, last)/2;
                // odd number of swaps
                if(swap&1)  parity *= -1;
                // even number of swaps, no change needed
                return false;
            }
        }
    }

    template<class RdmaccIt, typename T>
    RdmaccIt binary_search(RdmaccIt beg, RdmaccIt end, const T &val)
    {
        RdmaccIt not_found = end;

        RdmaccIt mid = beg + (end - beg)/2;
        while(mid != end && *mid != val)
        {
            if(*mid > val)
                end = mid;
            else
                beg = mid + 1;
            mid = beg + (end - beg)/2;
        }

        if(mid == end)
            return not_found;

        return mid;
    }

    // Comp(*it, val) output should be defined as
    // =0 : ==
    // >0 : >
    // <0 : <
    template<class RdmaccIt, typename T, class Comp>
    RdmaccIt binary_search(RdmaccIt beg, RdmaccIt end, const T &val, Comp comp)
    {
        RdmaccIt not_found = end;

        RdmaccIt mid = beg + (end - beg)/2;
        while(mid != end && comp(*mid,val) != 0)
        {
            if(comp(*mid, val) > 0)
                end = mid;
            else
                beg = mid + 1;
            mid = beg + (end - beg)/2;
        }

        if(mid == end)
            return not_found;

        return mid;
    }

    template<class RdmaccIt, typename T>
    std::pair<RdmaccIt, RdmaccIt> binary_search_interval(RdmaccIt beg,
                                                         RdmaccIt end,
                                                         const T &val)
    {
        RdmaccIt first = beg, last = end;
        RdmaccIt mid = beg + (end - beg)/2;
        while(mid != end)
        {
            if(*mid == val)
                return std::make_pair(mid, mid);
            else if(*mid > val)
                end = mid;
            else
                beg = mid + 1;
            mid = beg + (end - beg)/2;
        }

        if(mid == last)
            return std::make_pair(last, last);

        if(*mid < val) {
            return std::make_pair(mid, mid + 1);
        } else {
            if(mid == first)
                return std::make_pair(last, last);
            return std::make_pair(mid - 1, mid);
        }

    }

    template<class RdmaccIt, typename T, class Compare>
    std::pair<RdmaccIt, RdmaccIt> binary_search_interval(RdmaccIt beg,
                                                         RdmaccIt end,
                                                         const T &val,
                                                         Compare comp)
    {
        RdmaccIt first = beg, last = end;
        RdmaccIt mid = beg + (end - beg)/2;
        while(mid != end)
        {
            if(comp(*mid, val) == 0)
                return std::make_pair(mid, mid);
            else if(comp(*mid, val) > 0)
                end = mid;
            else
                beg = mid + 1;
            mid = beg + (end - beg)/2;
        }

        if(mid == last)
            return std::make_pair(last, last);

        if(comp(*mid, val) < 0) {
            return std::make_pair(mid, mid + 1);
        } else {
            if(mid == first)
                return std::make_pair(last, last);
            return std::make_pair(mid - 1, mid);
        }
    }

    template<class RdmaccIt, typename T>
    RdmaccIt binary_search_close_less(RdmaccIt beg, RdmaccIt end, const T &val)
    {
        if(beg == end)
            return end;
        if(*(end - 1) <= val)
            return end - 1;

        RdmaccIt first = beg, last = end;
        RdmaccIt mid = beg + (end - beg)/2;
        while(mid != end)
        {
            if(*mid == val)
                return mid;
            if(*mid > val)
                end = mid;
            else
                beg = mid + 1;
            mid = beg + (end - beg)/2;
        }

        if(*mid < val) {
            return mid;
        } else {
            if(mid == first)
                return last;
            return mid - 1;
        }
    }
};

#endif
