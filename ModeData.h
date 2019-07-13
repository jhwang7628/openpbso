#ifndef __MODE_DATA_H__
#define __MODE_DATA_H__
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
//##############################################################################
// ModeData: Simple data structure for storing modal displacements and
//           frequencies
//##############################################################################
template<typename REAL>
struct ModeData {
public:
    // Eigen-values produced by modal analysis
    std::vector<REAL> _omegaSquared;
    std::vector<std::vector<REAL>> _modes;

    int _N_modesAudible = -1;
    REAL _freqThresCache = 22100.;
    REAL _densityCache = -1;

public:
    inline std::vector<REAL> &mode( int modeIndex )
    {return _modes.at(modeIndex);}
    inline const std::vector<REAL> &mode( int modeIndex ) const
    {return _modes.at(modeIndex);}
    inline REAL omegaSquared( int modeIndex ) const
    {return _omegaSquared.at(modeIndex);}
    inline int numModes() const
    {return _omegaSquared.size();}
    inline int numDOF() const
    {return ( numModes() > 0 ) ? _modes.at(0).size() : 0;}

    void read(const char *filename);
    void write(const char *filename) const;
    void printAllFrequency(const REAL &density) const;
    int numModesAudible(const REAL &density, const REAL &audibleFreq);

    friend std::ostream &operator <<(std::ostream &os, const ModeData &data) {
        os << "------------------------------------------------\n"
           << "Struct ModeData\n"
           << "------------------------------------------------\n"
           << " number of modes : " << data.numModes() << "\n"
           << " number of DOF   : " << data.numDOF() << "\n"
           << "------------------------------------------------"
           << std::flush;
        return os;
    }
};
//##############################################################################
// Reads mode data from a file
//##############################################################################
template<typename REAL>
void ModeData<REAL>::read(const char *filename) {
    std::ifstream fin( filename, std::ios::binary );
    assert(fin.good() && "cannot open file for reading modes");

    // Read the size of the problem and the number of modes
    int nDOF, nModes;
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
//##############################################################################
// Writes mode data to a file
//##############################################################################
template<typename REAL>
void ModeData<REAL>::write( const char *filename ) const
{
    std::ofstream fout( filename, std::ios::binary );
    assert(fout.good() && "cannot open file for writing modes");
    int nModes = _omegaSquared.size();
    int nDOF;
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
//##############################################################################
template<typename REAL>
void ModeData<REAL>::printAllFrequency(const REAL &density) const
{
    typedef typename std::vector<REAL>::const_iterator Iterator;
    int count=0;
    for (Iterator it =_omegaSquared.begin();
                  it!=_omegaSquared.end();
                  ++it, count++)
        printf("Mode %u: %f Hz\n", count, sqrt((*it)/density)/(2.*M_PI));
}
//##############################################################################
template<typename REAL>
int ModeData<REAL>::numModesAudible(
    const REAL &density, const REAL &audibleFreq) {
    // use cache
    if (density == _densityCache &&
        _freqThresCache == audibleFreq &&
        _N_modesAudible >= 0) {
        return _N_modesAudible;
    }
    auto Freq = [&](const REAL os)->REAL {
        return sqrt(os/density)/(2.*M_PI);
    };
    if (_omegaSquared.size() == 0 || Freq(_omegaSquared.at(0)) > audibleFreq) {
        return 0;
    }
    if (Freq(_omegaSquared.at(_omegaSquared.size()-1)) <= audibleFreq) {
        return _omegaSquared.size();
    }
    int ii;
    for (ii=0; ii<_omegaSquared.size(); ++ii) {
        if (Freq(_omegaSquared.at(ii)) > audibleFreq) {
            break;
        }
    }
    _N_modesAudible = ii;
    _densityCache = density;
    _freqThresCache = audibleFreq;
    return _N_modesAudible;
}
#endif // __MODE_DATA_H__
