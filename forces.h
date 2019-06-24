#ifndef FORCES_H
#define FORCES_H
#include <random>
//##############################################################################
enum class ForceType {
    PointForce = 0,
    GaussianForce = 1,
    AutoregressiveForce = 2
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
class Force {
public:
    virtual bool Add(Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) = 0;
    virtual ~Force() = default;
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
class PointForce : public Force<T,BUF_SIZE>{
private:
    bool used = false;
public:
    bool Add(Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) override;
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
class GaussianForce : public Force<T,BUF_SIZE>{
private:
    T _width; // in time units
    int _widthSamples;
    int _count = 0;
    int _center;
    int _cutoff = 5;
public:
    GaussianForce(const T width)
        : _width(width) {
        _widthSamples = std::max(1, (int)(_width/1000000.*SAMPLE_RATE));
        _center = (int)((_cutoff-0.5) * _widthSamples);
    }
    bool Add(Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) override;
};
//##############################################################################
template<typename T>
struct AutoregressiveForceParam {
    std::vector<T> a = {0.783, 0.116};
    T sigma = 0.00148;
    T mu = 0.142;
};
//##############################################################################
// Paper:
//  Pai et al. Scanning Physical Interaction Behavior of 3D Objects, 2001
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
class AutoregressiveForce : public Force<T,BUF_SIZE>{
private:
    std::vector<T> _buf;
    const int _bufLen;
    int _bufIdx = 0;
    std::vector<T> _a;
    T _sigma;
    T _mu;
    std::default_random_engine _generator;
    std::normal_distribution<T> _distribution; // mean 0.0, stddev 1.0
    T GetMuEffective();
public:
    AutoregressiveForce()
        : _buf{0,0,0}, _bufLen(_buf.size()),
          _a{0.783,0.116}, _sigma(0.00148), _mu(0.142) {
    }
    bool Add(Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) override;
    void SetParam(const AutoregressiveForceParam<T> &param);
};
//##############################################################################
template<typename T, int BUF_SIZE>
bool PointForce<T, BUF_SIZE>::Add(
    Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) {
    if (used) {
        return false;
    }
    forceSpread(0) += 1.;
    used = true;
    return true;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool GaussianForce<T, BUF_SIZE>::Add(
    Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) {
    if (_width == 0 || _count >= _cutoff*2*_widthSamples) {
        return false;
    }
    for (int ii=0; ii<BUF_SIZE; ++ii) {
        const T p =
            -(T)0.5 * std::pow((T)(_count+ii-_center)/(T)_widthSamples, 2);
        forceSpread(ii) += std::exp(p);
    }
    _count += BUF_SIZE;
    return true;
}
//##############################################################################
template<typename T, int BUF_SIZE>
T AutoregressiveForce<T, BUF_SIZE>::GetMuEffective() {
    T mu_tilde = (T)0.0;
    for (int ii=0; ii<2; ++ii) {
        mu_tilde += _a.at(ii) * _buf.at((_bufIdx+_bufLen-ii-1)%_bufLen);
    }
    mu_tilde += _sigma * _distribution(_generator);
    _buf.at(_bufIdx) = mu_tilde;
    _bufIdx = (_bufIdx+1) % _bufLen;
    return _mu + mu_tilde;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool AutoregressiveForce<T, BUF_SIZE>::Add(
    Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) {
    T mu_e;
    for (int ii=0; ii<BUF_SIZE; ++ii) {
        mu_e = GetMuEffective();
        forceSpread(ii) += mu_e;
    }
    return true;
}
//##############################################################################
template<typename T, int BUF_SIZE>
void AutoregressiveForce<T, BUF_SIZE>::SetParam(
    const AutoregressiveForceParam<T> &param) {
    _buf = {0,0,0};
    _a = param.a;
    _sigma = param.sigma;
    _mu = param.mu;
}
//##############################################################################
#endif
