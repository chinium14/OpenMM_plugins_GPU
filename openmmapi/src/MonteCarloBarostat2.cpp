/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 *                                                                            *
 * This code is part of the plugins for OpenMM and inherited from the         *
 * the MonteCarloBarostat.cpp by Peter Eastman.                               *
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

#include "openmm/MonteCarloBarostat2.h"
#include "openmm/internal/MonteCarloBarostatImpl2.h"
#include <ctime>

using namespace OpenMM;

MonteCarloBarostat2::MonteCarloBarostat2(double defaultPressure, double temperature, Vec3& pressure3D, double surfaceTension, int frequency):
        defaultPressure(defaultPressure), temperature(temperature),
        pressure3D(pressure3D), surfaceTension(surfaceTension), frequency(frequency) {
    setRandomNumberSeed((int) time(NULL));
}

ForceImpl* MonteCarloBarostat2::createImpl() const {
    return new MonteCarloBarostatImpl2(*this);
}

