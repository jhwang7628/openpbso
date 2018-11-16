#ifndef MODAL_ODE_SOLVER_H
#define MODAL_ODE_SOLVER_H 
#include <memory> 
#include <cmath>
#include "config.h"
#include "ModalMaterial.h"

//##############################################################################
// This class constructs and solves the decoupled modal ODE for linear
// modal analysis. Namely,
//
// d^2q + 2 xi omega dq + omega^2 q = Q / m. 
//
// This is equation 8 from DyRT paper [James 2002]. The naming convention and
// the solution procedure using the efficient IIR filter follows from this
// paper as well. 
//##############################################################################
class ModalODESolver
{
    public: 
        typedef std::shared_ptr<ModalMaterial> ModalMaterialPtr; 

    private: 
        ModalMaterialPtr _material; 
        REAL _omegaSquared; // as in eq. 8
        REAL _omega; // as in eq. 8
        //REAL _qNew; // stores q^(k-1) in eq. 19
        //REAL _qOld; // stores q^(k-2) in eq. 20
        REAL _timeStepSize; // h in eq. 19
        REAL _epsilon; // stroes epsilon_i = exp(-xi_i omega_i h) in eq. 19
        REAL _theta; // stroes theta_i = w_di h in eq. 19
        REAL _gamma; // stores gamma_i = arcsin(xi_i) in eq. 19
        REAL _time; 

        // following are cached floats to speed things up
        REAL _2_epsilon_cosTheta;  // coeff in first term in eq. 19.
        REAL _epsilon_squared; // coeff in second term in eq. 19.
        REAL _coeff_Q_i; // coeff in third term in eq. 19. 

        bool _initialized;

    public:
        ModalODESolver()
            : _time(0.0), _initialized(false)
        {}

        inline ModalMaterialPtr GetModalMaterial(){return _material;}
        inline REAL GetODECurrentTime(){return _time;}
        inline void SetODECurrentTime(const REAL &time){_time = time;}
        inline void SetModalMaterial(ModalMaterialPtr &material){_material = material;} 
        inline void SetModalFrequency(REAL &omegaSquared){_omegaSquared = omegaSquared; _omega = sqrt(omegaSquared);}
        inline REAL GetCoefficient_qNew(){return _2_epsilon_cosTheta;} 
        inline REAL GetCoefficient_qOld(){return -_epsilon_squared;} 
        inline REAL GetCoefficient_Q(){return _coeff_Q_i;} 
        void Initialize(ModalMaterialPtr &material, const REAL &omegaSquared, const REAL &timeStepSize); 
        REAL StepSystem(REAL qOld, REAL qNew, const REAL &Q); // inplace computes q^(k) from q^(k-1) and q^(k-2) and Q_i
};

#endif
