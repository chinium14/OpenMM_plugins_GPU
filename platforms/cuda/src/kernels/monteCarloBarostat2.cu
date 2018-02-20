/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is the cuda version of atom position rescaling code for the           *
 * ansiotropical barostat as part of the OpenMM plugins.                      *
 *                                                                            *
 * Portions copyright (c) 2013 the University of Michigan and the Authors.    *
 * Authors: Shuai Wei                                                         *
 * Contributors: Charles L. Brooks III and Michael Garrahan                   *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

extern "C" __global__ void scalePositions2(double4 scale, int numMolecules, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4* __restrict__ posq,
        const int* __restrict__ moleculeAtoms, const int* __restrict__ moleculeStartIndex) {
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < numMolecules; index += blockDim.x*gridDim.x) {
        int first = moleculeStartIndex[index];
        int last = moleculeStartIndex[index+1];
        int numAtoms = last-first;

        // Find the center of each molecule.

        real3 center = make_real3(0, 0, 0);
        for (int atom = first; atom < last; atom++) {
            real4 pos = posq[moleculeAtoms[atom]];
            center.x += pos.x;
            center.y += pos.y;
            center.z += pos.z;
        }
        real invNumAtoms = RECIP(numAtoms);
        center.x *= invNumAtoms;
        center.y *= invNumAtoms;
        center.z *= invNumAtoms;

        // Move it into the first periodic box.
        int xcell;
        int ycell;
        int zcell;
        xcell = (int) floor(center.x*invPeriodicBoxSize.x);
        ycell = (int) floor(center.y*invPeriodicBoxSize.y);
        zcell = (int) floor(center.z*invPeriodicBoxSize.z);

        real3 delta = make_real3(xcell*periodicBoxSize.x, ycell*periodicBoxSize.y, zcell*periodicBoxSize.z);

        center.x -= delta.x;
        center.y -= delta.y;
        center.z -= delta.z;

        // Now scale the position of the molecule center.
        
        delta.x = center.x*(scale.x-1)-delta.x;
        delta.y = center.y*(scale.y-1)-delta.y;
        delta.z = center.z*(scale.z-1)-delta.z;
        for (int atom = first; atom < last; atom++) {
            real4 pos = posq[moleculeAtoms[atom]];
            pos.x += delta.x;
            pos.y += delta.y;
            pos.z += delta.z;
            posq[moleculeAtoms[atom]] = pos;
        }
    }
}

