// This file is part of openpbso, an open-source library for physics-based sound
//
// Copyright (C) 2018 Jui-Hsien Wang <juiwang@alumni.stanford.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/. */
#ifndef FFAT_MAP_SERIALIZE
#define FFAT_MAP_SERIALIZE
#include <fstream>
#include "io.h"
#include "ffat_solver.h"
#include "ffat_map.pb.h"
//##############################################################################
namespace Gpu_Wavesolver {
//##############################################################################
// (de)serialize vectors
# define SERIALIZE_VEC(__vec_ptr, __vec_size, __data_in) \
    for (int ii=0; ii<(int)(__vec_size); ++ii) { \
        __vec_ptr->add_item(__data_in[ii]); \
    }
# define DESERIALIZE_VEC(__vec_ptr, __vec_size, __data_out, __resize) \
    if (__resize) { \
        __data_out.resize(__vec_size); \
    } else { \
        assert((int)__data_out.size() == __vec_size && \
               "fixed data size inconsistent"); \
    } \
    for (int ii=0; ii<(int)(__vec_size); ++ii) { \
        __data_out[ii] = __vec_ptr.item(ii); \
    }
// (de)serialize column-major eigen matrix
# define SERIALIZE_EMAT(__mat_ptr, __vec_ptr, __data_in) \
    { \
        const int __N_i = __data_in.cols(); \
        const int __N_j = __data_in.rows(); \
        for (int ii=0; ii<__N_i; ++ii) { \
            __vec_ptr = __mat_ptr->add_item(); \
            for (int jj=0; jj<__N_j; ++jj) { \
                __vec_ptr->add_item(__data_in(jj,ii)); \
            } \
        } \
    }
# define DESERIALIZE_EMAT(__mat_ptr, __mat_col, __mat_row, __data_out) \
    { \
        __data_out.resize(__mat_row, __mat_col); \
        for (int ii=0; ii<__mat_col; ++ii) { \
            const auto &__vec_ptr = __mat_ptr.item(ii); \
            for (int jj=0; jj<__mat_row; ++jj) { \
                __data_out(jj,ii) = __vec_ptr.item(jj); \
            } \
        } \
    }
//##############################################################################
// This serialize class save/load the minimal amount of fields to run GetMapVal
// function for runtime ffat map query. The necessary fields are listed below.
// The class should be used as a wrapper to protobuf data.
//
// Needed data:
//   FFAT_Map<T,3>
//     _shells (FFAT_Map<T,1>)
//       - Intersect
//       - Interpolate
//       - GetDataQuadStride
//     _is_compressed
//     _compressed_Psi or _Psi
//     _k
//     _center
//     modeId
//
//   FFAT_Map<T,1>
//     _center (same as FFAT_Map<T,3>)
//     _bboxLow
//     _bboxTop
//     _lowCorners
//     _N_elements
//     _cellSize
//     _strides
//##############################################################################
struct FFAT_Map_Serialize_Double {
    static void Save(const char *filename, const FFAT_Map<double,3> &map);
    static void Load(const char *filename, FFAT_Map<double,3> &map);
    static std::map<int,FFAT_Map<double,3>> *LoadAll(const char *dirname);
    template<typename T>
    static bool MatchBits(const T *data1, const T *data2, const int size);
    static bool Check(
        const FFAT_Map<double,3> &map1, const FFAT_Map<double,3> &map2);
};
//##############################################################################
void FFAT_Map_Serialize_Double::Save(
    const char *filename, const FFAT_Map<double,3> &map_3_in) {
    ffat_map::ffat_map_double ffat_map;
    ffat_map::ffat_map_t_3 *map_3 = ffat_map.mutable_map();
    ffat_map::ffat_map_t_1 *map_1 = map_3->mutable_shells();
    const FFAT_Map<double,1> &map_1_in = map_3_in._shells.at(2);

    ffat_map::vec *vec_ptr;
    ffat_map::mat *mat_ptr;
    ffat_map::vec_i *vec_i_ptr;
    ffat_map::mat_i *mat_i_ptr;
    int N_i, N_j;
    // set FFAT_Map<T,1> fields
    // set cellSize
    map_1->set_cellsize(map_1_in._cellSize);
    // set lowCorners
    mat_ptr = map_1->mutable_lowcorners();
    N_i = map_1_in._lowCorners.size();
    for (int ii=0; ii<N_i; ++ii) {
        vec_ptr = mat_ptr->add_item();
        N_j = map_1_in._lowCorners[ii].size();
        for (int jj=0; jj<N_j; ++jj) {
            vec_ptr->add_item(map_1_in._lowCorners[ii][jj]);
        }
    }
    // set N_elements
    mat_i_ptr = map_1->mutable_n_elements();
    N_i = map_1_in._N_elements.size();
    for (int ii=0; ii<N_i; ++ii) {
        vec_i_ptr = mat_i_ptr->add_item();
        vec_i_ptr->add_item(map_1_in._N_elements[ii].first);
        vec_i_ptr->add_item(map_1_in._N_elements[ii].second);
    }
    // set strides
    SERIALIZE_VEC(map_1->mutable_strides(),
                  map_1_in._strides.size(),
                  map_1_in._strides);
    // set center
    SERIALIZE_VEC(map_1->mutable_center(),
                  map_1_in._center.size(),
                  map_1_in._center);
    // set bboxLow
    SERIALIZE_VEC(map_1->mutable_bboxlow(),
                  map_1_in._bboxLow.size(),
                  map_1_in._bboxLow);
    // set bboxTop
    SERIALIZE_VEC(map_1->mutable_bboxtop(),
                  map_1_in._bboxTop.size(),
                  map_1_in._bboxTop);

    // set FFAT_Map<T,3> fields
    // set k
    map_3->set_k(map_3_in._k);
    // set center
    SERIALIZE_VEC(map_3->mutable_center(),
                  map_3_in._center.size(),
                  map_3_in._center);
    // set is_compressed
    map_3->set_is_compressed(map_3_in._is_compressed);
    if (map_3_in._is_compressed) {
        // serialize _compressed_Psi
        SERIALIZE_EMAT(map_3->mutable_psi(),
            vec_ptr,
            map_3_in._compressed_Psi);
    } else {
        // serialize _Psi
        SERIALIZE_EMAT(map_3->mutable_psi(),
            vec_ptr,
            map_3_in._Psi);
    }
    map_3->set_modeid(map_3_in.modeId);
    std::ofstream stream(filename, std::ios::binary);
    ffat_map.SerializeToOstream(&stream);
    stream.close();
}
//##############################################################################
void FFAT_Map_Serialize_Double::Load(
    const char *filename, FFAT_Map<double,3> &map_3_out) {
    // parse from stream
    ffat_map::ffat_map_double ffat_map;
    std::ifstream stream(filename, std::ios::binary);
    ffat_map.ParseFromIstream(&stream);
    // fill the data structure
    const ffat_map::ffat_map_t_3 &map_3 = ffat_map.map();
    const ffat_map::ffat_map_t_1 &map_1 = map_3.shells();
    int N_i, N_j;
    // parse FFAT_Map<T,1> fields
    FFAT_Map<double,1> map_1_out;
    // parse cellsize
    map_1_out._cellSize = map_1.cellsize();
    // parse lowcorners
    N_i = map_1.lowcorners().item_size();
    N_j = 3;
    map_1_out._lowCorners.resize(N_i);
    for (int ii=0; ii<N_i; ++ii) {
        for (int jj=0; jj<N_j; ++jj) {
            map_1_out._lowCorners[ii][jj] =
                map_1.lowcorners().item(ii).item(jj);
        }
    }
    // parse n_elements
    N_i = map_1.n_elements().item_size();
    N_j = 2;
    map_1_out._N_elements.resize(N_i);
    for (int ii=0; ii<N_i; ++ii) {
        map_1_out._N_elements[ii] = std::pair<int,int>(
            map_1.n_elements().item(ii).item(0),
            map_1.n_elements().item(ii).item(1));
    }
    // parse strides
    DESERIALIZE_VEC(
        map_1.strides(),
        map_1.strides().item_size(),
        map_1_out._strides,
        true);
    // parse center
    DESERIALIZE_VEC(
        map_1.center(),
        map_1.center().item_size(),
        map_1_out._center,
        false);
    // parse bboxlow
    DESERIALIZE_VEC(
        map_1.bboxlow(),
        map_1.bboxlow().item_size(),
        map_1_out._bboxLow,
        false);
    // parse bboxtop
    DESERIALIZE_VEC(
        map_1.bboxtop(),
        map_1.bboxtop().item_size(),
        map_1_out._bboxTop,
        false);
    // parse FFAT_Map<T,3> fields
    // parse k
    map_3_out._k = map_3.k();
    // parse center
    DESERIALIZE_VEC(
        map_3.center(),
        map_3.center().item_size(),
        map_3_out._center,
        false);
    // parse shells
    map_3_out._shells.resize(3);
    map_3_out._shells[2] = std::move(map_1_out);
    // parse is_compressed
    map_3_out._is_compressed = map_3.is_compressed();
    // parse psi
    if (map_3.is_compressed()) {
        // parse compressed psi
        DESERIALIZE_EMAT(
            map_3.psi(),
            map_3.psi().item_size(),
            map_3.psi().item(0).item_size(),
            map_3_out._compressed_Psi);
    } else {
        // parse psi
        DESERIALIZE_EMAT(
            map_3.psi(),
            map_3.psi().item_size(),
            map_3.psi().item(0).item_size(),
            map_3_out._Psi);
    }
    map_3_out.modeId = map_3.modeid();
}
//##############################################################################
template<typename T>
bool FFAT_Map_Serialize_Double::MatchBits(
    const T *data1, const T *data2, const int size) {
    for (int ii=0; ii<size; ++ii) {
        if (data1[ii] != data2[ii]) {
            return false;
        }
    }
    return true;
}
//##############################################################################
std::map<int,FFAT_Map<double,3>> *FFAT_Map_Serialize_Double::LoadAll(
    const char *dirname) {
    std::vector<std::string> filenames;
    ListDirFiles(dirname, filenames, ".fatcube");
    std::map<int,FFAT_Map<double,3>> *map =
        new std::map<int,FFAT_Map<double,3>>();
    for (const auto &filename : filenames) {
        FFAT_Map<double,3> map_;
        FFAT_Map_Serialize_Double::Load(filename.c_str(), map_);
        (*map)[map_.modeId] = map_;
    }
    return map;
}
//##############################################################################
bool FFAT_Map_Serialize_Double::Check(
    const FFAT_Map<double,3> &map1, const FFAT_Map<double,3> &map2) {
    bool match = true;
    const FFAT_Map<double,1> &map1_1 = map1._shells.at(2);
    const FFAT_Map<double,1> &map2_1 = map2._shells.at(2);
    match &= MatchBits(
        &(map1_1._cellSize), &(map2_1._cellSize),
        1);
    match &= MatchBits(
        map1_1._lowCorners[0].data(), map2_1._lowCorners[0].data(),
        map1_1._lowCorners.size()*3);
    match &= MatchBits(
        &(map1_1._N_elements[0].first), &(map2_1._N_elements[0].first),
        map1_1._N_elements.size()*2);
    match &= MatchBits(
        &(map1_1._strides[0]), &(map2_1._strides[0]),
        map1_1._strides.size());
    match &= MatchBits(
        map1_1._center.data(), map2_1._center.data(),
        3);
    match &= MatchBits(
        map1_1._bboxLow.data(), map2_1._bboxLow.data(),
        3);
    match &= MatchBits(
        map1_1._bboxTop.data(), map2_1._bboxTop.data(),
        3);
    match &= MatchBits(
        &(map1._k), &(map2._k),
        1);
    match &= MatchBits(
        map1._center.data(), map2._center.data(),
        3);
    match &= MatchBits(
        &(map1._is_compressed), &(map2._is_compressed),
        1);
    if (map1._is_compressed) {
        match &= MatchBits(
            map1._compressed_Psi.data(), map2._compressed_Psi.data(),
            map1._compressed_Psi.rows()*map1._compressed_Psi.cols());
    } else {
        match &= MatchBits(
            map1._Psi.data(), map2._Psi.data(),
            map1._Psi.rows()*map1._Psi.cols());
    }
    match &= MatchBits(
        &(map1.modeId), &(map2.modeId),
        1);
    return match;
}
//##############################################################################
typedef FFAT_Map_Serialize_Double FFAT_Map_Serialize; // default to double
//##############################################################################
}; // namespace Gpu_Wavesolver
//##############################################################################
#endif
