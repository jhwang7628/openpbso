///////////////////////////////////////////////////////////////////////////////
// ModeData.h: Simple data structure for storing modal displacements and
//             frequencies
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __MODE_DATA_H__
#define __MODE_DATA_H__

#include "config.h"
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
struct ModeData {
    public:
        // Eigen-values produced by modal analysis
        std::vector<REAL>                    _omegaSquared;

        std::vector<std::vector<REAL> >      _modes;

    public:
        std::vector<REAL>       &mode( int modeIndex )
                                 {
                                    return _modes.at(modeIndex);
                                 }

        const std::vector<REAL> &mode( int modeIndex ) const
                                 {
                                    return _modes.at(modeIndex);
                                 }

        REAL                     omegaSquared( int modeIndex ) const
                                 {
                                    return _omegaSquared.at(modeIndex);
                                 }

        int                      numModes() const
                                 {
                                     return _omegaSquared.size();
                                 }

        int                      numDOF() const
                                 {
                                     return ( numModes() > 0 ) ? _modes.at(0).size() : 0;
                                 }

        void                     read( const char *filename );
        void                     write( const char *filename ) const;
        void                     printAllFrequency(const REAL &density) const;

    friend std::ostream &operator <<(std::ostream &os, const ModeData &data); 
};

#endif // __MODE_DATA_H__
