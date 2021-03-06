/* Portions copyright (c) 2010 Stanford University and Simbios.
 * Contributors: Peter Eastman
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <string.h>

#include <openmm/reference/SimTKOpenMMUtilities.h>
#include "ReferenceMonteCarloBarostat2.h"

using namespace std;
using namespace OpenMM;

ReferenceMonteCarloBarostat2::ReferenceMonteCarloBarostat2(int numAtoms, const vector<vector<int> >& molecules) : molecules(molecules) {
    savedAtomPositions[0].resize(numAtoms);
    savedAtomPositions[1].resize(numAtoms);
    savedAtomPositions[2].resize(numAtoms);
}

ReferenceMonteCarloBarostat2::~ReferenceMonteCarloBarostat2( ) {
}

/**
 * Apply the barostat at the start of a time step.
 *
 * @param atomPositions      atom positions
 * @param boxSize            the periodic box dimensions
 * @param scale              the factor by which to scale atom positions
 */
void ReferenceMonteCarloBarostat2::applyBarostat(vector<RealVec>& atomPositions, const RealVec& boxSize, Vec3& scale) {
    int numAtoms = savedAtomPositions[0].size();
    for (int i = 0; i < numAtoms; i++)
        for (int j = 0; j < 3; j++)
            savedAtomPositions[j][i] = atomPositions[i][j];

    // Loop over molecules.

    for (int i = 0; i < (int) molecules.size(); i++) {
        // Find the molecule center.

        RealOpenMM pos[3] = {0, 0, 0};
        for (int j = 0; j < (int) molecules[i].size(); j++) {
            RealVec& atomPos = atomPositions[molecules[i][j]];
            pos[0] += atomPos[0];
            pos[1] += atomPos[1];
            pos[2] += atomPos[2];
        }
        pos[0] /= molecules[i].size();
        pos[1] /= molecules[i].size();
        pos[2] /= molecules[i].size();

        int xcell;
        int ycell;
        int zcell;
        // Move it into the first periodic box.
        xcell = (int) floor(pos[0]/boxSize[0]);
        ycell = (int) floor(pos[1]/boxSize[1]);
        zcell = (int) floor(pos[2]/boxSize[2]);

        RealOpenMM dx = xcell*boxSize[0];
        RealOpenMM dy = ycell*boxSize[1];
        RealOpenMM dz = zcell*boxSize[2];
        pos[0] -= dx;
        pos[1] -= dy;
        pos[2] -= dz;

        // Now scale the position of the molecule center.

        dx = pos[0]*(scale[0]-1)-dx;
        dy = pos[1]*(scale[1]-1)-dy;
        dz = pos[2]*(scale[2]-1)-dz;
        for (int j = 0; j < (int) molecules[i].size(); j++) {
            RealVec& atomPos = atomPositions[molecules[i][j]];
            atomPos[0] += dx;
            atomPos[1] += dy;
            atomPos[2] += dz;
        }
    }
}

/**
 * Restore atom positions to what they were before applyBarostat() was called.
 *
 * @param atomPositions      atom positions
 */
void ReferenceMonteCarloBarostat2::restorePositions(vector<RealVec>& atomPositions) {
    int numAtoms = savedAtomPositions[0].size();
    for (int i = 0; i < numAtoms; i++)
        for (int j = 0; j < 3; j++)
            atomPositions[i][j] = savedAtomPositions[j][i];
}

