// This file is part of openpbso, an open-source library for physics-based sound
//
// Copyright (C) 2018 Jui-Hsien Wang <juiwang@alumni.stanford.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/. */
#include "io.h"
#include <string>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
//##############################################################################
namespace Gpu_Wavesolver {
//##############################################################################
void ListDirFiles(const char *dirname, std::vector<std::string> &names,
    const char *contains) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dirname)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            std::string f = dirname + std::string("/") + std::string(ent->d_name);
            if (IsFile(f.c_str()) && (ent->d_name[0] != '.') &&
                (contains && f.find(std::string(contains)) != std::string::npos))
                names.push_back(f);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
    }
}
//##############################################################################
bool IsFile(const char *path) {
    // seems to fail sometimes..
    //struct stat path_stat;
    //stat(path, &path_stat);
    //return S_ISREG(path_stat.st_mode);
    // ref: https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
    struct stat buffer;
    return (stat (path, &buffer) == 0);
}
//##############################################################################
std::string Basename(const std::string &path) {
    auto start = path.find_last_of("/");
    return path.substr(start+1);
}
//##############################################################################
};
//##############################################################################
