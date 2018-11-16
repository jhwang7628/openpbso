/******************************************************************************
 *  File: config.h
 *
 *  This file is part of isostuffer
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
#ifndef CONFIG_H
#define CONFIG_H

/*
 * Determine to use hash_map or unordered_map
 */
#ifdef __INTEL_COMPILER
#   if (__INTEL_COMPILER >= 1100 ) && (__INTEL_COMPILER <= 1210)
#       define USE_HASH_MAP 1
#   else
#       define USE_UNORDERED_MAP 1
#   endif
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#   if (__GNUC__ == 4 && __GNUC_MINOR__ < 3)
#       define USE_HASH_MAP 1
#   else
#       define USE_UNORDERED_MAP 1
#   endif
#else
#   error ERROR: The version of compiler is too low to support hash_map/unordered_map
#endif

#define REAL            double
#define REAL_INFINITY   1e50

#include <limits>
const REAL  EPS   = 1e-10;
const REAL  F_LOW = std::numeric_limits<REAL>::lowest(); 
const REAL  F_MAX = std::numeric_limits<REAL>::max(); 

// Usefull definition
#define SDUMP(x)	" " << #x << "=[ " << x << " ] "
#define COUT_SDUMP(x) std::cout << SDUMP(x) << std::endl

#endif
