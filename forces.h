#ifndef FORCES_H
#define FORCES_H
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
        std::cout << "width samples = " << _widthSamples << std::endl; // FIXME debug
        _center = (int)((_cutoff-0.5) * _widthSamples);
    }
    bool Add(Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) override;
};
//##############################################################################
template<typename T, int BUF_SIZE=FRAMES_PER_BUFFER>
class AutoregressiveForce : public Force<T,BUF_SIZE>{
public:
    bool Add(Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) override;
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
        //// FIXME debug START
        //std::cout << "ii = " << ii << std::endl;
        //std::cout << (_count+ii-_center) << std::endl;
        //std::cout << _widthSamples << std::endl;
        //std::cout << std::pow((T)(_count+ii-_center)/(T)_widthSamples, 2) << std::endl;
        //std::cout << p << std::endl;
        //std::cout << std::exp(p) << std::endl;
        //// FIXME debug STOP
        std::cout << std::exp(p) << std::endl;
        forceSpread(ii) += std::exp(p);
    }
    _count += BUF_SIZE;
    return true;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool AutoregressiveForce<T, BUF_SIZE>::Add(
    Eigen::Matrix<T,BUF_SIZE,1> &forceSpread) {
    return false;
}
//##############################################################################
#endif
