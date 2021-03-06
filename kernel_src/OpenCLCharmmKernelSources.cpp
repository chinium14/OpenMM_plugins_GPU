#include "OpenCLCharmmKernelSources.h"

using namespace std;
using namespace OpenMM;
const string OpenCLCharmmKernelSources::monteCarloBarostat2 = "/**\n"
" * Scale the particle positions.\n"
" */\n"
"\n"
"__kernel void scalePositions2(real4 scale, int numMolecules, real4 periodicBoxSize, real4 invPeriodicBoxSize, __global real4* restrict posq,\n"
"        __global const int* restrict moleculeAtoms, __global const int* restrict moleculeStartIndex) {\n"
"    for (int index = get_global_id(0); index < numMolecules; index += get_global_size(0)) {\n"
"        int first = moleculeStartIndex[index];\n"
"        int last = moleculeStartIndex[index+1];\n"
"        int numAtoms = last-first;\n"
"\n"
"        // Find the center of each molecule.\n"
"\n"
"        real4 center = (real4) 0;\n"
"        for (int atom = first; atom < last; atom++)\n"
"            center += posq[moleculeAtoms[atom]];\n"
"        center /= (real) numAtoms;\n"
"\n"
"        // Move it into the first periodic box.\n"
"\n"
"        int xcell = (int) floor(center.x*invPeriodicBoxSize.x);\n"
"        int ycell = (int) floor(center.y*invPeriodicBoxSize.y);\n"
"        int zcell = (int) floor(center.z*invPeriodicBoxSize.z);\n"
"        real4 delta = (real4) (xcell*periodicBoxSize.x, ycell*periodicBoxSize.y, zcell*periodicBoxSize.z, 0);\n"
"        center -= delta;\n"
"\n"
"        // Now scale the position of the molecule center.\n"
"\n"
"        delta.x = center.x*(scale.x-1)-delta.x;\n"
"        delta.y = center.y*(scale.y-1)-delta.y;\n"
"        delta.z = center.z*(scale.z-1)-delta.z;\n"
"        delta.w = center.w*(scale.w-1)-delta.w;\n"
"        for (int atom = first; atom < last; atom++) {\n"
"            real4 pos = posq[moleculeAtoms[atom]];\n"
"            pos.xyz += delta.xyz;\n"
"            posq[moleculeAtoms[atom]] = pos;\n"
"        }\n"
"    }\n"
"}\n"
"";

