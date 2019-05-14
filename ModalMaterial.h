#ifndef MODAL_MATERIAL_H
#define MODAL_MATERIAL_H
#include <fstream>
#include <sstream>
#include "config.h"
//##############################################################################
// This struct stores and computes all the material paramters needed for modal
// analysis. The naming convention follows DyRT paper [James 2002]. In
// particular, IIR filter equation is useful to time-step the modal equations.
// (eq. 11, 19).
//##############################################################################
template<typename REAL>
struct ModalMaterial {
    std::string name;
    REAL alpha;
    REAL beta;
    REAL density; // assuming we always have constant, uniform density
    REAL poissonRatio;
    REAL youngsModulus;

    // cached fields
    REAL inverseDensity;
    inline REAL xi(const REAL &omega_i) const
    {return 0.5*(alpha/omega_i + beta*omega_i);} // eq.10, xi = [0,1]
    inline REAL omega_di(const REAL &omega_i) const
    {return omega_i*sqrt(1.0 - pow(xi(omega_i),2));} // eq.12.

    static ModalMaterial *Read(const char *filename) {
        ModalMaterial *material = nullptr;
        std::ifstream stream(filename);
        if (stream) {
            material = new ModalMaterial();
            material->name = filename;
            std::string line;
            while(std::getline(stream, line)) {
                if (line[0] != '#') {
                    break;
                }
            }
            std::istringstream iss(line);
            iss >> material->density;
            iss >> material->youngsModulus;
            iss >> material->poissonRatio;
            iss >> material->alpha;
            iss >> material->beta;
        }
        return material;
    }
};
#endif
