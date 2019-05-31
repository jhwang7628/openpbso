#ifndef FORCES_H
#define FORCES_H
//##############################################################################
enum class ForceType {
    PointForce = 0,
    GaussianForce = 1,
    AutoregressiveForce = 2
};
//##############################################################################
template<typename T, int BUF_SIZE>
class Force {
    virtual bool Generate() = 0;
};
//##############################################################################
template<typename T, int BUF_SIZE>
class PointForce : public Force<T,BUF_SIZE>{
    bool Generate() override;
};
//##############################################################################
template<typename T, int BUF_SIZE>
class GaussianForce : public Force<T,BUF_SIZE>{
private:
    T _width;
public:
    GaussianForce(const T width)
        : _width(width) {
    }
    bool Generate() override;
};
//##############################################################################
template<typename T, int BUF_SIZE>
class AutoregressiveForce : public Force<T,BUF_SIZE>{
    bool Generate() override;
};
//##############################################################################
template<typename T, int BUF_SIZE>
bool PointForce<T, BUF_SIZE>::Generate() {
    return true;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool GaussianForce<T, BUF_SIZE>::Generate() {
    return true;
}
//##############################################################################
template<typename T, int BUF_SIZE>
bool AutoregressiveForce<T, BUF_SIZE>::Generate() {
    return true;
}
//##############################################################################
#endif
