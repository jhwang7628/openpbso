#ifndef FORCES_H
#define FORCES_H
//##############################################################################
template<typename T, int BUF_SIZE>
class Force {
    virtual bool Generate() = 0;
};

//##############################################################################
template<typename T, int BUF_SIZE>
class GaussianForce : public Force<T,BUF_SIZE>{
    bool Generate() override;
};
//##############################################################################
template<typename T, int BUF_SIZE>
class AutoregressiveForce : public Force<T,BUF_SIZE>{
    bool Generate() override;
};
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
