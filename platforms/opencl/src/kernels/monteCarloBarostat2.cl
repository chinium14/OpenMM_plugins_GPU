/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is the opencl version of atom position rescaling code for the         *
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


__kernel void scalePositions2(real4 scale, int numMolecules, real4 periodicBoxSize, real4 invPeriodicBoxSize, __global real4* restrict posq,
        __global const int* restrict moleculeAtoms, __global const int* restrict moleculeStartIndex) {
    for (int index = get_global_id(0); index < numMolecules; index += get_global_size(0)) {
        int first = moleculeStartIndex[index];
        int last = moleculeStartIndex[index+1];
        int numAtoms = last-first;

        // Find the center of each molecule.

        real4 center = (real4) 0;
        for (int atom = first; atom < last; atom++)
            center += posq[moleculeAtoms[atom]];
        center /= (real) numAtoms;

        // Move it into the first periodic box.

        int xcell = (int) floor(center.x*invPeriodicBoxSize.x);
        int ycell = (int) floor(center.y*invPeriodicBoxSize.y);
        int zcell = (int) floor(center.z*invPeriodicBoxSize.z);
        real4 delta = (real4) (xcell*periodicBoxSize.x, ycell*periodicBoxSize.y, zcell*periodicBoxSize.z, 0);
        center -= delta;

        // Now scale the position of the molecule center.

        delta.x = center.x*(scale.x-1)-delta.x;
        delta.y = center.y*(scale.y-1)-delta.y;
        delta.z = center.z*(scale.z-1)-delta.z;
        delta.w = center.w*(scale.w-1)-delta.w;
        for (int atom = first; atom < last; atom++) {
            real4 pos = posq[moleculeAtoms[atom]];
            pos.xyz += delta.xyz;
            posq[moleculeAtoms[atom]] = pos;
        }
    }
}

