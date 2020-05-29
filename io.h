// This file is part of openpbso, an open-source library for physics-based sound
//
// Copyright (C) 2018 Jui-Hsien Wang <juiwang@alumni.stanford.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/. */
#ifndef IO_H
#define IO_H
#include <fstream>
#include <complex>
#include <vector>
#include <string>
#include <iomanip>
#include "Eigen/Dense"
//##############################################################################
namespace Gpu_Wavesolver {
//##############################################################################
void ListDirFiles(const char *dirname, std::vector<std::string> &names,
    const char *contains=nullptr);
bool IsFile(const char *filename);
std::string Basename(const std::string &path);
//##############################################################################
template<typename T_i, typename T_o>
void ReadComplexVector(
    const char *filename,
    Eigen::Matrix<std::complex<T_o>, Eigen::Dynamic, 1> &p,
    const bool binary) {
    if (binary) {
        std::ifstream stream(filename, std::ios::binary);
        assert(stream && "file not exist");
        int count;
        stream.read((char*)&count, sizeof(int));
        Eigen::Matrix<T_i,Eigen::Dynamic,Eigen::Dynamic> tmp;
        tmp.resize(2, count/2);
        stream.read((char*)tmp.data(), sizeof(T_i)*count);
        p.resize(count/2);
        for (int ii=0; ii<count/2; ++ii) {
            p(ii).real((T_o)tmp(0, ii));
            p(ii).imag((T_o)tmp(1, ii));
        }
    }
    else {
        std::ifstream stream(filename);
        assert(stream && "file not exist");
        std::string line;
        int count = 0;
        while(std::getline(stream, line)) {
            count ++;
        }
        stream.close();
        stream.open(filename);
        p.resize(count);
        std::complex<T_o> p_i;
        T_o a,b;
        for (int ii=0; ii<count; ++ii) {
            std::getline(stream, line);
            std::istringstream iss(line);
            iss >> a >> b;
            p_i.real(a);
            p_i.imag(b);
            p(ii) = p_i;
        }
    }
}
//##############################################################################
template<typename T_i>
void WriteComplexVector(
    const char *filename,
    const Eigen::Matrix<std::complex<T_i>, Eigen::Dynamic, 1> &p,
    const bool binary) {
    if (binary) {
        std::ofstream stream(filename, std::ios::binary);
        assert(stream && "can't open file");
        int count = p.rows()*2;
        stream.write((char*)&count, sizeof(int));
        for (int ii=0; ii<p.rows(); ++ii) {
            stream.write((char*)&(p(ii,0)), sizeof(T_i)*2);
        }
    }
    else {
        std::ofstream stream(filename);
        assert(stream && "can't open file");
        for (int ii=0; ii<p.rows(); ++ii) {
            stream << std::fixed << std::setprecision(16);
            stream << p(ii).real() << " " << p(ii).imag() << std::endl;
        }
        stream.close();
    }
}
//##############################################################################
}; // namespace Gpu_Wavesolver
//##############################################################################
#endif
