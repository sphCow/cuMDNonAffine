//
// Created by parswa_arch on 12/4/15.
//

#ifndef SEQMD_THERMOSTATBDP_H
#define SEQMD_THERMOSTATBDP_H

//#include "MDBase.h"
//class MDBase;
#include <iostream>
#include <cmath>

using namespace std;

class ThermostatBDP {
private:
    double taut;
    double ran1();
    double gasdev();
    double gamdev(const int ia);
    double resamplekin_sumnoises(int nn);
public:
    void set(double taut_);
    double resamplekin(double kk,double sigma, unsigned long ndeg);

};


#endif //SEQMD_THERMOSTATBDP_H
