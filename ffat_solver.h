#ifndef FFAT_SOLVER_H
#define FFAT_SOLVER_H
#include <iostream>
#include <complex>
#include <map>
#include "Eigen/Dense"
#include "io.h"
#include "igl/serialize.h"
#include "igl/opengl/glfw/Viewer.h"
#include "igl/write_triangle_mesh.h"
//#define USE_OPENCV
#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#endif
//##############################################################################
namespace Gpu_Wavesolver {
    //##############################################################################
    // Forward declaration
struct FFAT_Map_Serialize_Double;
typedef FFAT_Map_Serialize_Double FFAT_Map_Serialize;
//##############################################################################
template<typename T, int M> class FFAT_Solver { /*not implemented*/ };
//##############################################################################
template<typename T, int M> class FFAT_Map { /*not implemented*/ };
//##############################################################################
template<typename T>
class FFAT_Solver<T,1> {
public:
    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> FFAT_MatrixXd;
    typedef Eigen::Matrix<T,Eigen::Dynamic,1> FFAT_VectorXd;
    typedef Eigen::Matrix<std::complex<T>,Eigen::Dynamic,1> FFAT_VectorXcd;
    typedef Eigen::Matrix<T,3,1> FFAT_Vector3d;

    // The model used is the 1-map FFAT map introduced in harmonic shells paper.
    // Namely,
    //
    // p(x) ~ h_0(kr) Psi(theta, phi),
    //
    // where h_0 is the 0-th order spherical hankel function of the first kind.
    // h_0 = - i exp(-ikr)/(kr)
    //
    // Inverting the expression is trivial:
    // Psi(theta, phi) = p(x) / h_0(kr)
    static void Solve(
        const T k,
        const FFAT_MatrixXd &X,
        const FFAT_Vector3d &x_0,
        const FFAT_VectorXcd &P,
        FFAT_VectorXcd &Psi);

    // Inversion of the Solve method.
    static void Reconstruct(
        const T k,
        const FFAT_Vector3d &X,
        const FFAT_Vector3d &x_0,
        const std::complex<T> &Psi,
        std::complex<T> &P);
};
//##############################################################################
template<typename T>
class FFAT_Map<T,1> : public igl::Serializable {
public:
    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> FFAT_MatrixXd;
    typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> FFAT_MatrixXi;
    typedef Eigen::Matrix<std::complex<T>,Eigen::Dynamic,Eigen::Dynamic> FFAT_MatrixXcd;
    typedef Eigen::Matrix<T,Eigen::Dynamic,1> FFAT_VectorXd;
    typedef Eigen::Matrix<int,Eigen::Dynamic,1> FFAT_VectorXi;
    typedef Eigen::Matrix<std::complex<T>,Eigen::Dynamic,1> FFAT_VectorXcd;
    typedef Eigen::Matrix<T,3,1> FFAT_Vector3d;
    typedef Eigen::Matrix<int,3,1> FFAT_Vector3i;

    // Construct face mesh that looks like cube map.
    // This is a helper function for the wavesolver to output evaluation mesh on
    // a cube map.
    //
    // The faces are constructed in the order of
    // +x, -x, +y, -y, +z, -z
    // represented in triangles. The normals are oriented correctly.
    // The code also outputs number of triangles per cube face, which can be
    // used to parse the obj file. Note that the triangles are always in pair
    // of the underlying gpu wavesolver grid. Therefore, if necessary they can
    // be combined into quads easily.
    static void CubemapMesh(
        const Eigen::Vector3i &bboxLow_r,
        const Eigen::Vector3i &bboxTop_r,
        const T cellSize,
        const Eigen::Matrix<T,3,1> gridLowCorner,
        const Eigen::Vector3i dim,
        std::vector<Eigen::Matrix<T,3,1>> &V,
        std::vector<Eigen::Vector3i> &F,
        std::vector<int> &dataIndices,
        std::vector<std::pair<int,int>> &N_elements);
    // Note: This save method was observed to not be portable from my Linux
    // machine to my Mac machine.
    static void Save(const char *filename, const FFAT_Map<T,1> &map);
    static void Load(const char *filename, FFAT_Map<T,1> &map);
    static std::map<int,FFAT_Map<T,1>> *LoadAll(const char *dirname);
    // convert the "cube" map to cube map -- one that has equal pxl in each dim
    static FFAT_Map<T,1> *ResampleToUniformCube(
        const FFAT_Map<T,1> &map,
        const FFAT_Vector3d &center,
        const T cellSize,
        const int dim);

    FFAT_Map() = default;
    FFAT_Map(
        const int &modeId,
        const T cellSize,
        const Eigen::Matrix<T,Eigen::Dynamic,3> &V,
        const std::vector<std::pair<int,int>> &N_elements);
    inline const std::pair<int,int> &GetNElements(const int &mapInd0) {
        return _N_elements.at(mapInd0);
    }
    inline FFAT_Vector3d GetCenter() const {
        return _center;
    }
    inline T GetCellSize() const {
        return _cellSize;
    }
    inline const std::vector<FFAT_VectorXcd> &GetData() const {
        return _A;
    }
    inline int GetMaxDim() const {
        int dim = -1;
        for (const auto &p : _N_elements) {
            dim = std::max(dim, p.first);
            dim = std::max(dim, p.second);
        }
        return dim;
    }
    inline int GetDataQuadStride(const FFAT_Vector3i &mapInd) const {
        return _strides.at(mapInd(0))
            + mapInd(1)*_N_elements.at(mapInd(0)).second + mapInd(2);
    }
    void InitSerialization();
    void Solve(
        const T &k,
        const FFAT_VectorXcd &dirichletPressure);
    void ConvertToTriMesh(
        FFAT_MatrixXd &V,
        FFAT_MatrixXi &F,
        FFAT_VectorXcd &A);
    void ConvertToImages(
        std::vector<FFAT_MatrixXcd> &A);

    // mapInd : internal indices (map_ind, u, v) = ([0,6), [0,Nx), [0,Ny])
    void Intersect(const FFAT_Vector3d &p,
                   FFAT_Vector3d &surfPoint,
                   FFAT_Vector3i &mapInd) const;
    void Interpolate(const FFAT_Vector3d &surfPoint,
                     const FFAT_Vector3i &nnMapInd,
                     std::vector<FFAT_Vector3i> &mapIndices,
                     std::vector<T> &coeffs) const;
    std::complex<T> GetMapVal(const FFAT_Vector3d &p) const;
    void QuadFromMapInd(const FFAT_Vector3i &mapInd,
                        FFAT_MatrixXd &V,
                        FFAT_MatrixXi &F);

    int modeId;
private:
    T _k = -1;
    T _cellSize;
    std::vector<FFAT_Vector3d> _lowCorners;
    std::vector<std::pair<int,int>> _N_elements;
    std::vector<int> _strides; //size = 6
    int _N_elements_total;
    std::vector<FFAT_VectorXcd> _A; // amplitudes of the transfer function
    FFAT_Vector3d _center;
    FFAT_Vector3d _bboxLow;
    FFAT_Vector3d _bboxTop;

friend class FFAT_Map<T,3>;
friend FFAT_Map_Serialize;
};
//##############################################################################
template<typename T>
class FFAT_Solver<T,3> {
public:
    typedef Eigen::Matrix<T,3,3> FFAT_Matrix3d;
    typedef Eigen::Matrix<T,-1,-1> FFAT_MatrixXd;
    typedef Eigen::Matrix<T,-1,1> FFAT_VectorXd;
    typedef Eigen::Matrix<std::complex<T>,-1,1> FFAT_VectorXcd;
    typedef Eigen::Matrix<std::complex<T>,-1,-1> FFAT_MatrixXcd;
    typedef Eigen::Matrix<T,3,1> FFAT_Vector3d;
    typedef Eigen::Matrix<T,3,1> FFAT_Vector3cd;

    // The model used is the 3-map amplitude FFAT map
    //
    // |p(x;k)|^2 = c1(theta,phi)/(kr) + c2(theta,phi)/(kr)^2
    //            + c3(theta,phi)/(kr)^3
    //
    // {c1,c2,c3} are all real-valued functions
    //
    // This can be derived from the complex-valued FFAT map used in Harmonic
    // Shells.
    //
    // Use the 'Solve' method to fit [c1,c2,c3] into Psi, and the 'Reconstruct'
    // method to evaluate the |p(x)|^2 expression listed above.

    // @param R X-by-3 radius values, i.e., each row is [r1, r2, r3] for that
    // direction
    // @param P X-by-3 complex-valued pressure samples, each row is for
    // [p(r1), p(r2), p(r3)]
    // @param Psi X-by-3 map values, each row is [c1,c2,c3]
    static void Solve(
        const T k,
        const FFAT_MatrixXd &R,
        const FFAT_MatrixXcd &P,
        FFAT_MatrixXd &Psi);

    // Inversion of the Solve method.
    // @return p(x;k), NOT |p|^2
    static T Reconstruct(
        const T k,
        const T r,
        const FFAT_Vector3d &Psi);

    static T Scaling(
        const T k,
        const FFAT_MatrixXd &R,
        const FFAT_MatrixXcd &P,
        FFAT_MatrixXd &Psi);
};
//##############################################################################
template<typename T>
class FFAT_Map<T,3> : public igl::Serializable {
public:
    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> FFAT_MatrixXd;
    typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> FFAT_MatrixXi;
    typedef Eigen::Matrix<std::complex<T>,Eigen::Dynamic,Eigen::Dynamic> FFAT_MatrixXcd;
    typedef Eigen::Matrix<T,Eigen::Dynamic,1> FFAT_VectorXd;
    typedef Eigen::Matrix<int,Eigen::Dynamic,1> FFAT_VectorXi;
    typedef Eigen::Matrix<std::complex<T>,Eigen::Dynamic,1> FFAT_VectorXcd;
    typedef Eigen::Matrix<T,3,1> FFAT_Vector3d;
    typedef Eigen::Matrix<int,3,1> FFAT_Vector3i;

    static void Save(const char *filename, const FFAT_Map<T,3> &map);
    static void Load(const char *filename, FFAT_Map<T,3> &map);
    static std::map<int,FFAT_Map<T,3>> *LoadAll(const char *dirname);
    static void ReadNElementsFile(const char *filename,
        std::vector<std::vector<std::pair<int,int>>> &N_elements);

    FFAT_Map() = default;
    FFAT_Map(
        const int &modeId,
        const T cellSize,
        const Eigen::Matrix<T,Eigen::Dynamic,3> &V,
        const std::vector<std::vector<std::pair<int,int>>> &N_elements);
    inline FFAT_Vector3d GetCenter() const {
        return _center;
    }
    inline T GetCellSize() const {
        return _cellSize;
    }
    inline const FFAT_MatrixXd &GetData() const {
        return _Psi;
    }
    void InitSerialization();
    void Solve(
        const T &k,
        const FFAT_VectorXcd &dirichletPressure,
        const bool &powerScaling = false);
    void ConvertToImages(
        std::vector<FFAT_MatrixXd> &A);
    T Compress(const char *output_template="tmp-%u-%u-amp.jpg",
        const int quality=65);
    T GetMapVal(const FFAT_Vector3d &p, const bool getCompressed=false) const;

    int modeId;
private:
    T _k = -1;
    T _cellSize;
    FFAT_Vector3d _center;
    std::vector<FFAT_Map<T,1>> _shells; // use to store the different shells
    std::vector<std::vector<std::pair<int,int>>> _N_elements;
    std::vector<int> _strides; // size is 3
    FFAT_MatrixXd _Psi;
    int _N_elements_total; // equals to the sum of all _N_elements
    int _N_directions; // equals to the sum of the first shell of _N_elements.

    // optional compression
    bool _is_compressed = false;
    FFAT_MatrixXd _compressed_Psi;
friend FFAT_Map_Serialize;
};
//##############################################################################
//##############################################################################
template<typename T>
void FFAT_Solver<T,1>::Solve(
    const T k,
    const FFAT_MatrixXd &X,
    const FFAT_Vector3d &x_0,
    const FFAT_VectorXcd &P,
    FFAT_VectorXcd &Psi) {
    FFAT_Vector3d dx;
    T kr;
    assert(X.rows() == P.size() && "Inconsistent size between X and P");
    Psi.resize(P.size());
    const std::complex<T> J(0,1);
    for (int ii=0; ii<X.rows(); ++ii) {
        dx = X.row(ii).transpose() - x_0;
        kr = k*dx.norm();
        Psi(ii) = P(ii) / (-J*std::exp(-J*kr)/kr);
    }
}
//##############################################################################
template<typename T>
void FFAT_Solver<T,1>::Reconstruct(
    const T k,
    const FFAT_Vector3d &X,
    const FFAT_Vector3d &x_0,
    const std::complex<T> &Psi,
    std::complex<T> &P) {
    FFAT_Vector3d dx;
    T kr;
    const std::complex<T> J(0,1);
    dx = X - x_0;
    kr = k*dx.norm();
    P = Psi * (-J*std::exp(-J*kr)/kr);
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::CubemapMesh(
    const Eigen::Vector3i &bboxLow_r,
    const Eigen::Vector3i &bboxTop_r,
    const T cellSize,
    const Eigen::Matrix<T,3,1> gridLowCorner,
    const Eigen::Vector3i dim,
    std::vector<Eigen::Matrix<T,3,1>> &V_vec,
    std::vector<Eigen::Vector3i> &F_vec,
    std::vector<int> &dataIndices,
    std::vector<std::pair<int,int>> &N_elements) {
    dataIndices.clear();
    N_elements.clear();
    typedef Eigen::Matrix<T,3,1> Vector3d;
    typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXd;
    typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXd;

    const MatrixXd offCenterCoord = (MatrixXd(4,2) <<
            -1, -1,
             1, -1,
             1,  1,
            -1,  1).finished()*cellSize/(T)2.0;
    const Eigen::MatrixXi offCenterFace = (Eigen::MatrixXi(2,3) <<
            1, 2, 3,
            3, 4, 1).finished().array()-1;
    const Eigen::MatrixXi offCenterFaceS = (Eigen::MatrixXi(2,3) <<
            1, 3, 2,
            3, 1, 4).finished().array()-1;
    const VectorXd ones = (VectorXd(4) << 1,1,1,1).finished();

    Vector3d cell_center;
    Eigen::Vector3i ijk;
    // +x, -x, +y, -y, +z, -z
    for (int face_idx=0; face_idx<6; ++face_idx) {
        const int dk = face_idx/2;
        const int sign = face_idx - dk*2 == 0 ? +1 : -1;
        int di = (dk+1)%3;
        int dj = (dk+2)%3;
        cell_center[dk] = gridLowCorner[dk] + cellSize*bboxLow_r[dk];
        if (sign == 1)
            cell_center[dk] += cellSize*(bboxTop_r[dk]-bboxLow_r[dk]+(T)1.0);
        const int kk = sign == 1 ? bboxTop_r[dk] : bboxLow_r[dk];
        ijk[dk] = kk;
        for (int ii=bboxLow_r[di]; ii<=bboxTop_r[di]; ++ii) {
            cell_center[di] = gridLowCorner[di] + ((T)0.5+ii)*cellSize;
            ijk[di] = ii;
            for (int jj=bboxLow_r[dj]; jj<=bboxTop_r[dj]; ++jj) {
                ijk[dj] = jj;
                cell_center[dj] = gridLowCorner[dj] + ((T)0.5+jj)*cellSize;
                MatrixXd vertices = MatrixXd(4,3);
                vertices.col(dk) = cell_center[dk]*ones;
                vertices.col(di) = cell_center[di]*ones + offCenterCoord.col(0);
                vertices.col(dj) = cell_center[dj]*ones + offCenterCoord.col(1);
                if (sign == 1) {
                    F_vec.push_back(offCenterFace.row(0).array()+V_vec.size());
                    F_vec.push_back(offCenterFace.row(1).array()+V_vec.size());
                } else {
                    F_vec.push_back(offCenterFaceS.row(0).array()+V_vec.size());
                    F_vec.push_back(offCenterFaceS.row(1).array()+V_vec.size());
                }
                dataIndices.push_back(ijk[2]*dim[0]*dim[1]+ijk[1]*dim[0]+ijk[0]);
                dataIndices.push_back(ijk[2]*dim[0]*dim[1]+ijk[1]*dim[0]+ijk[0]);
                for (int row=0; row<4; ++row) {
                    V_vec.push_back(vertices.row(row));
                }
            }
        }
        // note here I am pushing back the number of quads rather than tris
        N_elements.push_back(std::pair<int,int>((bboxTop_r[di]-bboxLow_r[di]+1),
                                                (bboxTop_r[dj]-bboxLow_r[dj]+1)));
    }
}
//##############################################################################
template<typename T>
FFAT_Map<T,1>::FFAT_Map(
    const int &modeId_,
    const T cellSize,
    const Eigen::Matrix<T,Eigen::Dynamic,3> &V,
    const std::vector<std::pair<int,int>> &N_elements) :
        modeId(modeId_), _cellSize(cellSize),
        _N_elements(N_elements) {
    assert(N_elements.size() == 6 && "N_elements wrong size");
    int sum = 0;
    for (const auto &p : N_elements) {
        const int N = p.first * p.second;
        // merge the tris to quads and skip every 1 data
        FFAT_VectorXd data;
        data.resize(N);
        FFAT_Vector3d corner = V.row(sum*4);
        _lowCorners.push_back(corner);
        _strides.push_back(sum);
        sum += N;
    }
    assert(_strides.size() == 6 && "wrong stride size");
    _N_elements_total = sum;
    _center << _lowCorners.at(0)(0) + _lowCorners.at(1)(0),
               _lowCorners.at(2)(1) + _lowCorners.at(3)(1),
               _lowCorners.at(4)(2) + _lowCorners.at(5)(2);
    _center /= (T)2.0;
    for (const FFAT_Vector3d &c : _lowCorners) {
        for (int jj=0; jj<3; ++jj) {
            _bboxLow(jj) = std::min(_bboxLow(jj), c(jj));
            _bboxTop(jj) = std::max(_bboxTop(jj), c(jj));
        }
    }
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::InitSerialization() {
    this->Add(modeId, "modeId");
    this->Add(_k, "k");
    this->Add(_cellSize, "cellSize");
    this->Add(_lowCorners, "lowCorners");
    this->Add(_N_elements, "N_elements");
    this->Add(_strides, "strides");
    this->Add(_N_elements_total, "N_elements_total");
    this->Add(_A, "A");
    this->Add(_center, "center");
    this->Add(_bboxLow, "bboxLow");
    this->Add(_bboxTop, "bboxTop");
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::Solve(const T &k, const FFAT_VectorXcd &dirichletPressure) {
    if (_k == k)
        return;
    // compute centroid positions from compact representation
    FFAT_MatrixXd X; // centroids
    X.resize(_N_elements_total, 3);
    int count = 0;
    for (int dd=0; dd<6; ++dd) {
        const FFAT_Vector3d &lowCorner = _lowCorners.at(dd);
        int di, dj, dk;
        dk = dd/2;
        di = (dk+1)%3;
        dj = (dk+2)%3;
        FFAT_Vector3d ijk;
        ijk(dk) = 0;
        for (int ii=0; ii<_N_elements.at(dd).first; ++ii) {
            ijk(di) = ii+0.5;
            for (int jj=0; jj<_N_elements.at(dd).second; ++jj) {
                ijk(dj) = jj+0.5;
                X.row(count) = lowCorner + ijk*_cellSize;
                ++ count;
            }
        }
    }
    // stride the dirichlet pressure since they were subdivided for tri mesh but
    // we are using quads here.
    assert(dirichletPressure.size() == X.rows()*2 &&
        "Dirichlet pressure wrong size. \
        It should be written from fdtd solver for the triangle cube mesh.");
    FFAT_VectorXcd P;
    P.resize(X.rows());
    for (int ii=0; ii<P.rows(); ++ii) {
        P(ii) = dirichletPressure(ii*2);
    }
    // compute the scaling and stores only the abs value.
    FFAT_VectorXcd Psi;
    FFAT_Solver<T,1>::Solve(k, X, _center, P, Psi);
    count = 0;
    for (const auto &p : _N_elements) {
        _A.push_back(Psi.segment(count, p.first*p.second));
        count += p.first*p.second;
    }
    // memorize wavenumber
    _k = k;
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::Save(const char *filename, const FFAT_Map<T,1> &map) {
    igl::serialize(map, "serial_map", filename, true);
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::Load(const char *filename, FFAT_Map<T,1> &map) {
    igl::deserialize(map, "serial_map", filename);
}
//##############################################################################
template<typename T>
std::map<int,FFAT_Map<T,1>> *FFAT_Map<T,1>::LoadAll(const char *dirname) {
    std::vector<std::string> filenames;
    ListDirFiles(dirname, filenames, ".fatcube");
    std::map<int,FFAT_Map<T,1>> *map = new std::map<int,FFAT_Map<T,1>>();
    for (const auto &filename : filenames) {
        FFAT_Map<T,1> map_;
        FFAT_Map<T,1>::Load(filename.c_str(), map_);
        (*map)[map_.modeId] = map_;
    }
    return map;
}
//##############################################################################
template<typename T>
FFAT_Map<T,1> *FFAT_Map<T,1>::ResampleToUniformCube(
    const FFAT_Map<T,1> &map,
    const FFAT_Vector3d &center,
    const T cellSize,
    const int dim) {
    assert(dim%2 == 0 && "Input odd number for uniform cube");
    FFAT_Map<T,1> *newMap_ = new FFAT_Map<T,1>();
    FFAT_Map<T,1> &newMap  = *newMap_;
    // get geometry
    newMap.modeId = map.modeId;
    newMap._cellSize = cellSize;
    newMap._center = center;
    newMap._N_elements_total = pow(dim,2)*6;
    for (int dd=0; dd<6; ++dd) {
        const int dk = dd/2;
        const int di = (dk+1)%3;
        const int dj = (dk+2)%3;
        const int nml = dd%2==0 ? +1 : -1;
        FFAT_Vector3d corner;
        if (nml == -1)
            corner[dk] = center[dk] - dim/2*cellSize;
        else
            corner[dk] = center[dk] + dim/2*cellSize;
        corner[di] = center[di] - dim/2*cellSize;
        corner[dj] = center[dj] - dim/2*cellSize;
        newMap._lowCorners.push_back(corner);
        newMap._N_elements.push_back({dim, dim});
    }
    for (const FFAT_Vector3d &c : newMap._lowCorners) {
        for (int jj=0; jj<3; ++jj) {
            newMap._bboxLow(jj) = std::min(newMap._bboxLow(jj), c(jj));
            newMap._bboxTop(jj) = std::max(newMap._bboxTop(jj), c(jj));
        }
    }
    // get data by evaluation the expansion
    newMap._k = map._k;
    newMap._A.resize(6);
    FFAT_MatrixXd X;
    FFAT_Vector3d X_v;
    X.resize(1,3);
    int count = 0;
    for (int dd=0; dd<6; ++dd) {
        const FFAT_Vector3d &lowCorner = newMap._lowCorners.at(dd);
        int di, dj, dk;
        dk = dd/2;
        di = (dk+1)%3;
        dj = (dk+2)%3;
        FFAT_Vector3d ijk;
        ijk(dk) = 0;
        newMap._A.at(dd).resize(dim*dim);
        for (int ii=0; ii<dim; ++ii) {
            ijk(di) = ii+0.5;
            for (int jj=0; jj<dim; ++jj) {
                ijk(dj) = jj+0.5;
                X.row(0) = lowCorner + ijk*newMap._cellSize;
                X_v = X.row(0);
                const std::complex<T> val = map.GetMapVal(X_v);
                FFAT_VectorXcd P;
                P.resize(1);
                P << val;
                FFAT_VectorXcd Psi;
                FFAT_Solver<T,1>::Solve(newMap._k, X, newMap._center, P, Psi);
                newMap._A.at(dd)(ii*dim+jj) = Psi(0);
                ++ count;
            }
        }
    }

    return newMap_;
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::ConvertToTriMesh(
    FFAT_MatrixXd &V,
    FFAT_MatrixXi &F,
    FFAT_VectorXcd &A) {
    int N_f = 0;
    for (int dd=0; dd<6; ++dd) {
        const auto &p = _N_elements.at(dd);
        N_f += p.first * p.second;
    }
    const FFAT_MatrixXd offCenterCoord = (FFAT_MatrixXd(4,2) <<
            -1, -1,
             1, -1,
             1,  1,
            -1,  1).finished()*_cellSize/(T)2.0;
    const FFAT_VectorXd ones = (FFAT_VectorXd(4) << 1,1,1,1).finished();
    const Eigen::MatrixXi offCenterFace = (Eigen::MatrixXi(2,3) <<
            1, 2, 3,
            3, 4, 1).finished().array()-1;
    const Eigen::MatrixXi offCenterFaceS = (Eigen::MatrixXi(2,3) <<
            1, 3, 2,
            3, 1, 4).finished().array()-1;
    V.resize(N_f*4, 3);
    F.resize(N_f*2, 3);
    A.resize(N_f*2);
    int count = 0;
    for (int dd=0; dd<6; ++dd) {
        const auto &p = _N_elements.at(dd);
        const FFAT_Vector3d &lowCorner = _lowCorners.at(dd);
        FFAT_Vector3d v, ijk, off;
        const int dim = dd/2;
        ijk(dim) = -0.5;
        off(dim) = 0;
        for (int ii=0; ii<p.first; ++ii) {
            for (int jj=0; jj<p.second; ++jj) {
                const int dk = dim;
                const int di = (dim+1)%3;
                const int dj = (dim+2)%3;
                ijk(di) = ii;
                ijk(dj) = jj;
                off(di) = 1;
                off(dj) = 1;
                v = lowCorner + _cellSize*(FFAT_Vector3d::Constant(0.5) + ijk);
                FFAT_MatrixXd vertices;
                vertices.resize(4,3);
                vertices.col(dk) = v[dk]*ones;
                vertices.col(di) = v[di]*ones + offCenterCoord.col(0);
                vertices.col(dj) = v[dj]*ones + offCenterCoord.col(1);
                const int offset = (count+ii*p.second+jj);
                V.block(4*offset+0, 0, 4, 3) = vertices;
                if (dd%2 == 0)
                    F.block(offset*2, 0, 2, 3) = offCenterFace.array()+ 4*offset;
                else
                    F.block(offset*2, 0, 2, 3) = offCenterFaceS.array()+4*offset;
                if (_A.size()==6) {
                    A(offset*2+0) = _A.at(dd)(ii*p.second+jj);
                    A(offset*2+1) = _A.at(dd)(ii*p.second+jj);
                }
            }
        }
        count += p.first * p.second;
    }
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::ConvertToImages(
    std::vector<FFAT_MatrixXcd> &A) {
    A.resize(6);
    for (int kk=0; kk<6; ++kk) {
        const auto &p = _N_elements.at(kk);
        A.at(kk).resize(p.first, p.second);
        // internally data is row-major, so first get them in tmp then assign
        for (int ii=0; ii<p.first; ++ii) {
            for (int jj=0; jj<p.second; ++jj) {
                A.at(kk)(ii,jj) = _A.at(kk)(ii*p.second+jj);
            }
        }
    }
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::Intersect(const FFAT_Vector3d &p,
                              FFAT_Vector3d &surfPoint,
                              FFAT_Vector3i &mapInd) const {
    // ray-triangle intersection follows 22raytracing-accel in Steve's lecture
    FFAT_Vector3d d = _center - p;
    FFAT_Vector3d t_min = (_bboxLow - p).array() / d.array();
    FFAT_Vector3d t_max = (_bboxTop - p).array() / d.array();
    FFAT_Vector3d t_enter = t_min.array().min(t_max.array());
    const T t_en = t_enter.maxCoeff();
    surfPoint = p + t_en*d;
    // figure out which face it is by simply the closest
    T minDist = std::numeric_limits<T>::max();
    for (int dd=0; dd<3; ++dd) {
        if (std::abs(_bboxLow(dd)-surfPoint(dd)) < minDist) {
            minDist = std::abs(_bboxLow(dd)-surfPoint(dd));
            mapInd(0) = dd*2+1;
        }
        if (std::abs(_bboxTop(dd)-surfPoint(dd)) < minDist) {
            minDist = std::abs(_bboxTop(dd)-surfPoint(dd));
            mapInd(0) = dd*2;
        }
    }
    int di,dj,dk;
    dk = mapInd(0)/2;
    di = (dk+1)%3;
    dj = (dk+2)%3;
    auto Clamp = [](const int x, const int l, const int h) {
        return std::min(std::max(x, l), h);
    };
    mapInd(1) = std::floor(
        (surfPoint(di) - _lowCorners.at(mapInd(0))(di))/_cellSize);
    mapInd(2) = std::floor(
        (surfPoint(dj) - _lowCorners.at(mapInd(0))(dj))/_cellSize);
    mapInd(1) = Clamp(mapInd(1), 0, _N_elements.at(mapInd(0)).first-1);
    mapInd(2) = Clamp(mapInd(2), 0, _N_elements.at(mapInd(0)).second-1);
}
//##############################################################################
// c01   tx     c11
// <---------->
//      ^     c
//   ty |
//      |
// c00  v       c10
//##############################################################################
template<typename T>
T BilinearInterp(const T &tx, const T &ty,
                 const T &c00, const T &c10, const T &c01, const T &c11) {
#if 0
    T  a = c00 * ((T)1 - tx) + c10 * tx;
    T  b = c01 * ((T)1 - tx) + c11 * tx;
    return a * ((T)1 - ty) + b * ty;
#else // alternative
	return ((T)1 - tx) * ((T)1 - ty) * c00 +
	        tx * ((T)1 - ty) * c10 +
	        ((T)1 - tx) * ty * c01 +
	        tx * ty * c11;
#endif
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::Interpolate(const FFAT_Vector3d &surfPoint,
                                const FFAT_Vector3i &nnMapInd,
                                std::vector<FFAT_Vector3i> &mapIndices,
                                std::vector<T> &coeffs) const {
    mapIndices.resize(4);
    coeffs.resize(4);
    FFAT_Vector3i &c00ind = mapIndices.at(0);
    FFAT_Vector3i &c10ind = mapIndices.at(1);
    FFAT_Vector3i &c01ind = mapIndices.at(2);
    FFAT_Vector3i &c11ind = mapIndices.at(3);
    int di,dj,dk;
    dk = nnMapInd(0)/2;
    di = (dk+1)%3;
    dj = (dk+2)%3;
    int x,y,xp,yp,Nx,Ny;
    T tx, ty;
    Nx = _N_elements.at(nnMapInd(0)).first;
    Ny = _N_elements.at(nnMapInd(0)).second;
    const FFAT_Vector3d &low = _lowCorners.at(nnMapInd(0));
    const T &h = _cellSize;
    T x_float = (surfPoint(di) - (low(di)+(T)0.5*h))/h;
    T y_float = (surfPoint(dj) - (low(dj)+(T)0.5*h))/h;
    x = (int)std::floor(x_float);
    y = (int)std::floor(y_float);
    //x = std::floor((surfPoint(di) - (low(di)+(T)0.5*h))/h);
    //y = std::floor((surfPoint(dj) - (low(dj)+(T)0.5*h))/h);
    if (x < 0) {
        x = 0;
        xp= 0;
        tx = 0;
    }
    else if (x >= 0 && x < Nx-1) {
        xp = x + 1;
        tx = x_float - (T)x;
    }
    else {
        x = Nx - 1;
        xp= Nx - 1;
        tx = 0;
    }
    if (y < 0) {
        y = 0;
        yp= 0;
        ty = 0;
    }
    else if (y >= 0 && y < Ny-1) {
        yp = y + 1;
        ty = y_float - (T)y;
    }
    else {
        y = Ny - 1;
        yp= Ny - 1;
        ty = 0;
    }
    tx = std::min(std::max(tx, 0.0), 1.0);
    ty = std::min(std::max(ty, 0.0), 1.0);
    assert(tx>=0 && tx<=1 && "tx out of range");
    assert(ty>=0 && ty<=1 && "ty out of range");
    c00ind << nnMapInd(0), x , y ;
    c10ind << nnMapInd(0), xp, y ;
    c01ind << nnMapInd(0), x , yp;
    c11ind << nnMapInd(0), xp, yp;
    coeffs = {((T)1-tx)*((T)1-ty),
                    tx *((T)1-ty),
              ((T)1-tx)*      ty ,
                    tx *      ty};
}
//##############################################################################
template<typename T>
std::complex<T> FFAT_Map<T,1>::GetMapVal(const FFAT_Vector3d &p) const {
    FFAT_Vector3i mapInd;
    FFAT_Vector3d surfPoint;
    Intersect(p, surfPoint, mapInd);
    std::complex<T> val;

    // nearest neighbor
    //std::complex<T> psi = _A.at(
    //    mapInd(0))(mapInd(1)*_N_elements.at(mapInd(0)).second+mapInd(2));
    //FFAT_Solver<T,1>::Reconstruct(
    //    _k,
    //    p,
    //    _center,
    //    psi,
    //    val);

    std::vector<FFAT_Vector3i> mapIndices;
    std::vector<T> coeffs;
    Interpolate(surfPoint, mapInd, mapIndices, coeffs);

    std::complex<T> psi = {0,0};
    for (int ii=0; ii<(int)mapIndices.size(); ++ii) {
        const FFAT_Vector3i &mi = mapIndices.at(ii);
        const T &co = coeffs.at(ii);
        psi += co*_A.at(
            mi(0))(mi(1)*_N_elements.at(mi(0)).second+mi(2));
    }
    FFAT_Solver<T,1>::Reconstruct(
        _k,
        p,
        _center,
        psi,
        val);

    return val;
}
//##############################################################################
template<typename T>
void FFAT_Map<T,1>::QuadFromMapInd(const FFAT_Vector3i &mapInd,
                                   FFAT_MatrixXd &V,
                                   FFAT_MatrixXi &F) {

    const FFAT_MatrixXd offCenterCoord = (FFAT_MatrixXd(4,2) <<
            -1, -1,
             1, -1,
             1,  1,
            -1,  1).finished()*_cellSize/(T)2.0;
    const FFAT_VectorXd ones = (FFAT_VectorXd(4) << 1,1,1,1).finished();
    int di, dj, dk;
    dk = mapInd(0)/2;
    di = (dk+1)%3;
    dj = (dk+2)%3;
    FFAT_Vector3d ijk;
    ijk(di) = mapInd(1) + 0.5;
    ijk(dj) = mapInd(2) + 0.5;
    ijk(dk) = 0;
    FFAT_Vector3d cell_center = _lowCorners.at(mapInd(0)) + ijk*_cellSize;
    V.resize(4,3);
    V.col(dk) = cell_center[dk]*ones;
    V.col(di) = cell_center[di]*ones + offCenterCoord.col(0);
    V.col(dj) = cell_center[dj]*ones + offCenterCoord.col(1);
    F = (FFAT_MatrixXi(2,3) <<
        1, 2, 3,
        3, 4, 1).finished().array()-1;
}
//##############################################################################
template<typename T>
void FFAT_Solver<T,3>::Solve(
    const T k,
    const FFAT_MatrixXd &R,
    const FFAT_MatrixXcd &P,
    FFAT_MatrixXd &Psi) {
    assert(R.rows() == P.rows() && "Inconsistent rows between R and P");
    assert(R.cols() == P.cols() && "Inconsistent cols between R and P");
    Psi.resize(P.rows(), 1);
    FFAT_MatrixXd basis;
    basis.resize(P.cols(),1);
    for (int ii=0; ii<P.rows(); ++ii) {
        const FFAT_VectorXd kr = R.row(ii)*k;
        basis.col(0) = (T)1./kr.array().pow(1);
        //const FFAT_Vector3d p2 = P.row(ii).array().abs2(); // squared norm
        FFAT_VectorXd p2 = P.row(ii).array().abs(); // Euclidean norm

        // SVD
        Eigen::JacobiSVD<FFAT_MatrixXd, Eigen::FullPivHouseholderQRPreconditioner> svd(basis,
                Eigen::ComputeFullU | Eigen::ComputeFullV);
        //Eigen::JacobiSVD<FFAT_Matrix3d> svd(basis);
        //T cond = svd.singularValues()(0)
        //        / svd.singularValues()(svd.singularValues().size()-1);
        Psi.row(ii) = svd.solve(p2);
    }
}
//##############################################################################
template<typename T>
T FFAT_Solver<T,3>::Reconstruct(
    const T k,
    const T r,
    const FFAT_Vector3d &Psi) {
    const T kr = k*r;
    return std::abs(Psi(0)/kr);
}
//##############################################################################
template<typename T>
T FFAT_Solver<T,3>::Scaling(
    const T k,
    const FFAT_MatrixXd &R,
    const FFAT_MatrixXcd &P,
    FFAT_MatrixXd &Psi) {
    FFAT_Vector3d dx;
    T kr;
    assert(R.rows() == P.rows() && "Inconsistent size between X and P");
    assert(R.rows() == Psi.rows() && "Inconsistent size between X and Psi");

    T numer=0, denom=0;
    for (int ii=0; ii<R.rows(); ++ii) {
        kr = k*R(ii,0);
        numer += pow(std::abs(P(ii,0)),2);
        denom += pow(Psi(ii,0)/kr,2);
    }
    const T scale = sqrt(numer/denom);
    for (int ii=0; ii<Psi.rows(); ++ii) {
        Psi.row(ii) *= scale;
    }
    return scale;
}
//##############################################################################
template<typename T>
FFAT_Map<T,3>::FFAT_Map(
    const int &modeId_,
    const T cellSize,
    const Eigen::Matrix<T,Eigen::Dynamic,3> &V,
    const std::vector<std::vector<std::pair<int,int>>> &N_elements) :
        modeId(modeId_), _cellSize(cellSize),
        _N_elements(N_elements) {
    const int N_shells = N_elements.size();
    assert(N_shells > 1 && "need more than 1 shell");

    // first compute a sum in each row
    std::vector<int> sums;
    _N_elements_total = 0;
    for (const auto &v : N_elements) {
        int sum = 0;
        for (const auto &p : v) {
            sum += p.first * p.second;
        }
        sums.push_back(sum);
        _strides.push_back(_N_elements_total);
        _N_elements_total += sum;
    }

    _shells.clear();
    int offset = 0;
    for (int ii=0; ii<N_shells; ++ii) {
        FFAT_Map<T,1> map(modeId_,
                          cellSize,
                          V.block(offset, 0, sums.at(ii)*4, 3),
                          N_elements.at(ii));
        _shells.push_back(map);
        offset += sums.at(ii)*4;
    }
    for (int ii=1; ii<N_shells; ++ii) {
        assert(
            (_shells.at(0).GetCenter() - _shells.at(ii).GetCenter()).norm() < 1E-6);
    }
    _center = _shells.at(2).GetCenter();
    _N_directions = 0;
    for (const auto p : _N_elements.at(2)) {
        _N_directions += p.first*p.second;
    }
}
//##############################################################################
template<typename T>
void FFAT_Map<T,3>::InitSerialization() {
    this->Add(modeId, "modeId");
    this->Add(_k, "k");
    this->Add(_cellSize, "cellSize");
    this->Add(_center, "center");
    this->Add(_shells, "maps");
    this->Add(_N_elements, "N_elements");
    this->Add(_strides, "strides");
    this->Add(_Psi, "Psi");
    this->Add(_N_elements_total, "N_elements_total");
    this->Add(_N_directions, "N_directions");
    this->Add(_is_compressed, "is_compressed");
    this->Add(_compressed_Psi, "compressed_Psi");
}
//##############################################################################
template<typename T>
void FFAT_Map<T,3>::Solve(const T &k, const FFAT_VectorXcd &dirichletPressure,
                          const bool &powerScaling) {
    if (_k == k)
        return;
    assert(_N_directions > 0 && "Solve started before proper initialization");
    assert(dirichletPressure.size() == 2*_N_elements_total &&
           "Dirichlet pressure wrong size.");
    const int N_shells = _shells.size();
    FFAT_MatrixXd R;
    FFAT_MatrixXcd P;
    R.resize(_N_directions, N_shells);
    P.resize(_N_directions, N_shells);
    const auto &N_elements = _N_elements.at(2);
    int offset = 0;
    FFAT_Vector3d ijk;
    for (int dd=0; dd<6; ++dd) {
        const int dk = dd/2;
        const int di = (dk+1)%3;
        const int dj = (dk+2)%3;
        const int dim1 = N_elements.at(dd).first;
        const int dim2 = N_elements.at(dd).second;
        ijk(dk) = 0;
        for (int ii=0; ii<dim1; ++ii) {
            ijk(di) = (T)0.5 + ii;
            for (int jj=0; jj<dim2; ++jj) {
                ijk(dj) = (T)0.5 + jj;

                const FFAT_Vector3d pos0 =
                    _shells.at(2)._lowCorners.at(dd) + ijk*_cellSize;
                FFAT_Vector3i pos0ind;
                pos0ind << dd, ii, jj;
                for (int ss=0; ss<N_shells; ++ss) {
                    FFAT_Vector3d pos;
                    FFAT_Vector3i posind;
                    // find surface point
                    _shells.at(ss).Intersect(pos0, pos, posind);
                    // compute radius using surface point
                    T &r = R(offset+ii*dim2+jj, ss);
                    r = (pos - _center).norm();
                    std::complex<T> &p = P(offset+ii*dim2+jj, ss);
                    p = {0,0};
                    std::vector<FFAT_Vector3i> mapIndices;
                    std::vector<T> coeffs;
                    _shells.at(ss).Interpolate(pos, posind, mapIndices, coeffs);
                    for (int kk=0; kk<(int)mapIndices.size(); ++kk) {
                        p += coeffs.at(kk)*dirichletPressure(
                            2*_strides.at(ss)
                           +2*_shells.at(ss).GetDataQuadStride(
                                mapIndices.at(kk)));
                    }
                    //// nearest neighbor sampling
                    //p = dirichletPressure(
                    //   2*_strides.at(ss)
                    //  +2*_shells.at(ss).GetDataQuadStride(posind));
                }
            }
        }
        offset += dim1*dim2;
    }
    assert(offset == P.rows() && "offset not equal to data");

    FFAT_Solver<T,3>::Solve(k, R, P, _Psi);
    if (powerScaling) {
        FFAT_Solver<T,3>::Scaling(k, R, P, _Psi);
    }

    // memorize wavenumber
    _k = k;
}
//##############################################################################
template<typename T>
void FFAT_Map<T,3>::Save(const char *filename, const FFAT_Map<T,3> &map) {
    igl::serialize(map, "serial_map_ch3", filename, true);
}
////##############################################################################
template<typename T>
void FFAT_Map<T,3>::Load(const char *filename, FFAT_Map<T,3> &map) {
    igl::deserialize(map, "serial_map_ch3", filename);
}
//##############################################################################
template<typename T>
std::map<int,FFAT_Map<T,3>> *FFAT_Map<T,3>::LoadAll(const char *dirname) {
    std::vector<std::string> filenames;
    ListDirFiles(dirname, filenames, ".fatcube");
    std::map<int,FFAT_Map<T,3>> *map = new std::map<int,FFAT_Map<T,3>>();
    for (const auto &filename : filenames) {
        FFAT_Map<T,3> map_;
        FFAT_Map<T,3>::Load(filename.c_str(), map_);
        (*map)[map_.modeId] = map_;
    }
    return map;
}
//##############################################################################
template<typename T>
void FFAT_Map<T,3>::ReadNElementsFile(const char *filename,
    std::vector<std::vector<std::pair<int,int>>> &N_elements) {
    std::ifstream stream(filename);
    assert(stream && "File not exist");
    std::string line;
    N_elements.clear();
    while (std::getline(stream, line)) {
        std::istringstream iss(line);
        std::vector<std::pair<int,int>> v(6);
        std::pair<int,int> p;
        for (int ii=0; ii<6; ++ii) {
            iss >> p.first >> p.second;
            v[ii] = p;
        }
        N_elements.push_back(v);
    }
}
//##############################################################################
template<typename T>
void FFAT_Map<T,3>::ConvertToImages(
    std::vector<FFAT_MatrixXd> &A) {
    A.resize(6);
    int offset = 0;
    for (int kk=0; kk<6; ++kk) {
        const auto &p = _N_elements.at(2).at(kk);
        A.at(kk).resize(p.first, p.second);
        // internally data is row-major, so first get them in tmp then assign
        for (int ii=0; ii<p.first; ++ii) {
            for (int jj=0; jj<p.second; ++jj) {
                A.at(kk)(ii,jj) = _Psi(offset+ii*p.second+jj);
            }
        }
        offset += p.first*p.second;
    }
}
//##############################################################################
template<typename T>
T FFAT_Map<T,3>::Compress(const char *output_template, const int quality) {
#ifdef USE_OPENCV
    std::vector<Eigen::MatrixXd> A;
    Eigen::MatrixXd A_amp;
    this->ConvertToImages(A);
    int offset = 0;
    _compressed_Psi.resize(_Psi.rows(), _Psi.cols());
    T maxAmp_global = -1;
    // uniformly normalize the ffat maps (wrt all faces)
    for (int ii=0; ii<(int)A.size(); ++ii) {
        maxAmp_global = std::max(maxAmp_global, A.at(ii).maxCoeff());
    }
    for (int ii=0; ii<(int)A.size(); ++ii) {
        A_amp = A.at(ii);
        T maxAmp = A_amp.maxCoeff();
//#define UNIFORM_NORMALIZE
#ifdef UNIFORM_NORMALIZE
        maxAmp = maxAmp_global;
#endif
        A_amp *= 255/maxAmp;
        cv::Mat data, data_s;
        cv::eigen2cv(A_amp, data);
        data.convertTo(data_s, CV_8U);
        // write and immediately read back to memory
        {
            char buf[512];
            snprintf(buf, 512, output_template, modeId, ii);
            //snprintf(buf, 512, "test-%u-%u-amp.jpg", pair.first, ii);
            std::vector<int> settings;
            settings.push_back(cv::IMWRITE_JPEG_QUALITY);
            settings.push_back(quality);
            cv::imwrite(buf, data_s, settings);
            data_s = cv::imread(buf, cv::IMREAD_GRAYSCALE);
        }
        data_s.convertTo(data, CV_64F);
        cv::cv2eigen(data, A_amp);
        A_amp *= maxAmp/255.;

        auto pair = _N_elements.at(2).at(ii);
        assert(A_amp.rows()*A_amp.cols() == pair.first*pair.second && "wrong dimension");
        for (int jj=0; jj<pair.first; ++jj) {
            for (int kk=0; kk<pair.second; ++kk) {
                _compressed_Psi(offset + jj*pair.second+kk) = A_amp(jj,kk);
            }
        }
        offset += pair.first*pair.second;
    }
    _is_compressed = true;
    return maxAmp_global;
#else
    assert(false && "Without OpenCV there's no way to do compression");
    return (T)0;
#endif
}
//##############################################################################
template<typename T>
T FFAT_Map<T,3>::GetMapVal(const FFAT_Vector3d &p,
    const bool getCompressed) const {
    if (getCompressed) {
        assert(_is_compressed &&
            "asking for compressed values without compression");
    }
    FFAT_Vector3d surfPoint;
    FFAT_Vector3i mapInd;
    _shells.at(2).Intersect(p, surfPoint, mapInd);

    // bilinear interpolation
    std::vector<FFAT_Vector3i> mapIndices;
    std::vector<T> coeffs;
    _shells.at(2).Interpolate(surfPoint, mapInd, mapIndices, coeffs);
    T val = 0;
    FFAT_Vector3d psi;
    psi.setZero();
    for (int kk=0; kk<(int)mapIndices.size(); ++kk) {
        const int idx = _shells.at(2).GetDataQuadStride(mapIndices.at(kk));
        if (getCompressed)
            psi(0) += coeffs.at(kk) * _compressed_Psi(idx,0);
        else
            psi(0) += coeffs.at(kk) * _Psi(idx,0);
    }
    val = FFAT_Solver<T,3>::Reconstruct(_k, (p-_center).norm(), psi);
    return val;

//    // nearest neighbor sampling
//    const int idx = _shells.at(0).GetDataQuadStride(mapInd);
//    FFAT_Vector3d psi;
//    psi << _Psi(idx,0), 0, 0;
//    return FFAT_Solver<T,3>::Reconstruct(
//        _k, (p-_center).norm(), psi);
}
//##############################################################################
}; // namespace Gpu_Wavesolver
//##############################################################################
#endif
