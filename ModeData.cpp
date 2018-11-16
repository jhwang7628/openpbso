///////////////////////////////////////////////////////////////////////////////
// ModeData.cpp: Simple data structure for storing modal displacements and
//             frequencies
//
///////////////////////////////////////////////////////////////////////////////

#include "ModeData.h"

#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Reads mode data from a file
///////////////////////////////////////////////////////////////////////////////
void ModeData::read( const char *filename )
{
    ifstream                 fin( filename, std::ios::binary );
    int                      nDOF;
    int                      nModes;

    if ( !fin.good() ) {
        cerr << "ModeData::read: Cannot open file " << filename << " to read" << endl;
        return;
    }

    // Read the size of the problem and the number of modes
    fin.read( (char *)&nDOF, sizeof(int) );
    fin.read( (char *)&nModes, sizeof(int) );

    // Read the eigenvalues
    _omegaSquared.resize( nModes );
    fin.read( (char *)_omegaSquared.data(), sizeof(REAL) * nModes );

    // Read the eigenvectors
    _modes.resize( nModes );
    for ( int i = 0; i < nModes; i++ ) {
        _modes[ i ].resize( nDOF );
        fin.read( (char *)_modes[ i ].data(), sizeof(REAL) * nDOF );
    }

    fin.close();
}

///////////////////////////////////////////////////////////////////////////////
// Writes mode data to a file
///////////////////////////////////////////////////////////////////////////////
void ModeData::write( const char *filename ) const
{
    ofstream                 fout( filename, std::ios::binary );
    int                      nModes = _omegaSquared.size();
    int                      nDOF;

    if ( !fout.good() ) {
        cerr << "ModeData::write: Cannot open file " << filename << " for writing" << endl;
        return;
    }

    if ( nModes <= 0 ) {
        cerr << "ModeData::write: Cannot write empty mode set" << endl;
        return;
    }

    nDOF = _modes[ 0 ].size();

    fout.write( (const char *)&nDOF, sizeof(int) );
    fout.write( (const char *)&nModes, sizeof(int) );

    // Write the eigenvalues
    fout.write( (const char *)_omegaSquared.data(), sizeof(REAL) * nModes );

    // Write the eigenvectors
    for ( int i = 0; i < nModes; i++ ) {
        fout.write( (const char *)_modes[ i ].data(), sizeof(REAL) * nDOF );
    }

    fout.close();
}

///////////////////////////////////////////////////////////////////////////////
void ModeData::printAllFrequency(const REAL &density) const
{
    typedef std::vector<REAL>::const_iterator Iterator; 
    int count=0;
    for (Iterator it=_omegaSquared.begin(); it!=_omegaSquared.end(); ++it, count++)
        printf("Mode %u: %f Hz\n", count, sqrt((*it)/density)); 
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
std::ostream &operator <<(std::ostream &os, const ModeData &data)
{
    os << "--------------------------------------------------------------------------------\n" 
       << "Struct ModeData\n" 
       << "--------------------------------------------------------------------------------\n"
       << " number of modes : " << data.numModes() << "\n"
       << " number of DOF   : " << data.numDOF() << "\n"
       << "--------------------------------------------------------------------------------" 
       << std::flush; 
    return os; 
}
