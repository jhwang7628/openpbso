#ifndef MODAL_INTEGRATOR_H
#define MODAL_INTEGRATOR_H
#include <vector>
#include "Eigen/Dense"
////////////////////////////////////////////////////////////////////////////////
// Solves the following equation using IIR filter
//
// \ddot{q} + a \dot{q} + b q = f,
//
// where a, b, q, f are N-vectors
////////////////////////////////////////////////////////////////////////////////
template<typename T>
class ModalIntegrator {
    public:
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> ModalVec;
    private:
        std::vector<ModalVec> _q;
        ModalVec _a;
        ModalVec _b;
        ModalVec _f;
        const T _h; // (constant) time step size
        int _q_curr_ptr = 0;

        // intermediate results
        ModalVec _c1;
        ModalVec _c2;
        ModalVec _c3;

    public:
        ModalIntegrator(const int N, const T h,
                        const ModalVec &a, const ModalVec &b);
        static ModalIntegrator<T> *Build(const T density,
                                         const std::vector<T> omegaSquared,
                                         const T alpha, const T beta, const T h);
        const ModalVec &Step(const ModalVec &Q);
        const ModalVec &Step();
};

template<typename T>
ModalIntegrator<T> *ModalIntegrator<T>::Build(const T density,
                                              const std::vector<T> omegaSquared,
                                              const T alpha, const T beta,
                                              const T h) {
    const int N = omegaSquared.size();
    ModalVec a, b;
    a.resize(N);
    b.resize(N);
    T xi, omega;
    for (int ii=0; ii<N; ++ii) {
        omega = sqrt(omegaSquared.at(ii)/density);
        xi = (T)0.5*(alpha/omega + beta*omega);
        a(ii) = (T)2*xi*omega;
        b(ii) = pow(omega, 2);
    }
    ModalIntegrator<T> *integrator = new ModalIntegrator<T>(N, h, a, b);
    return integrator;
}

template<typename T>
ModalIntegrator<T>::ModalIntegrator(const int N, const T h,
                                    const ModalVec &a, const ModalVec &b)
    : _h(h), _a(a), _b(b) {
    _q.resize(3);
    for (int ii=0; ii<3; ++ii)
        _q.at(ii).setZero(N);
    assert(_a.size() == N && "Vec a has wrong size");
    assert(_b.size() == N && "Vec b has wrong size");
    _f.resize(N);
    _c1.resize(N);
    _c2.resize(N);
    _c3.resize(N);

    // get coeff for IIR (convention in DyRT paper)
    T epsilon, theta, gamma, omega, omega_d;
    for (int ii=0; ii<N; ++ii) {
        epsilon = exp(-a(ii)/2*h);
        theta = h*sqrt(b(ii) - a(ii)*a(ii)/(T)4.0);
        gamma = std::asin(a(ii)/((T)2*sqrt(b(ii))));
        omega = sqrt(b(ii));
        omega_d = sqrt(b(ii) - pow(a(ii),2)/(T)4);

        _c1(ii) = (T)2*epsilon*cos(theta);
        _c2(ii) = -pow(epsilon,2);
        _c3(ii) = (T)2*(epsilon*cos(theta+gamma)-pow(epsilon,2)*cos((T)2*theta+gamma));
        _c3(ii) /= ((T)3*omega*omega_d);
        _c3(ii) *= 1E9; // arbitrary scaling
    }
}

template<typename T>
const typename ModalIntegrator<T>::ModalVec &ModalIntegrator<T>::Step(const ModalVec &Q) {
    ModalVec &q_k         = _q.at((_q_curr_ptr+1)%3);
    const ModalVec &q_km1 = _q.at((_q_curr_ptr  )%3);
    const ModalVec &q_km2 = _q.at((_q_curr_ptr+2)%3);
    if (Q.size() == _c3.size()) {
        q_k = _c1.cwiseProduct(q_km1) + _c2.cwiseProduct(q_km2)
            + _c3.cwiseProduct(Q);
    } else {
        // FIXME debug temporary fix
        ModalVec Q_;
        Q_.setZero(_c3.size());
        Q_.head(Q.size()) = Q;
        q_k = _c1.cwiseProduct(q_km1) + _c2.cwiseProduct(q_km2)
            + _c3.cwiseProduct(Q_);
    }
    _q_curr_ptr = (_q_curr_ptr + 1)%3;
    return q_k;
}

template<typename T>
const typename ModalIntegrator<T>::ModalVec &ModalIntegrator<T>::Step() {
    ModalVec &q_k         = _q.at((_q_curr_ptr+1)%3);
    const ModalVec &q_km1 = _q.at((_q_curr_ptr  )%3);
    const ModalVec &q_km2 = _q.at((_q_curr_ptr+2)%3);
    q_k = _c1.cwiseProduct(q_km1) + _c2.cwiseProduct(q_km2);
    _q_curr_ptr = (_q_curr_ptr + 1)%3;
    return q_k;
}
#endif
