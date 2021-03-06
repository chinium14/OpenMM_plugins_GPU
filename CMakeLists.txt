#---------------------------------------------------
# CHARMM OpenMM MC Barostat 2 Plugin
#
# Creates mcbarostat2 plugin library.
#
#   libmcbarostat2.so
#----------------------------------------------------

ADD_LIBRARY(mcbarostat2 SHARED
  platforms/reference/src/ReferenceCharmmKernelFactory.cpp
  platforms/reference/src/ReferenceCharmmKernels.cpp
  platforms/reference/src/ReferenceMonteCarloBarostat2.cpp
  wrappers/CharmmOpenMMFortranWrapper.cpp
  wrappers/CharmmOpenMMCWrapper.cpp
  openmmapi/src/MonteCarloBarostat2.cpp
  openmmapi/src/MonteCarloBarostatImpl2.cpp
  serialization/MonteCarloBarostat2SerializationProxyRegistration.cpp
  serialization/MonteCarloBarostat2Proxy.cpp)

target_include_directories(mcbarostat2 PRIVATE
  ${OPENMM_INCLUDE_DIRS}
  platforms/reference/include
  openmmapi/include
  wrappers
  omm_headers
  kernel_src)


if(APPLE)
  target_link_libraries(mcbarostat2 ${OPENMM_LIBRARIES})
  set_property(TARGET mcbarostat2
               PROPERTY MACOSX_RPATH ON)
endif()

set(MCBAROSTAT2_TARGS mcbarostat2)

if(CUDA_FOUND)
  add_library(mcbarostat2CUDA SHARED
    kernel_src/CudaCharmmKernelSources.cpp
    platforms/cuda/src/CudaCharmmKernelFactory.cpp
    platforms/cuda/src/CudaCharmmKernels.cpp)
  target_include_directories(mcbarostat2CUDA PRIVATE
    ${OPENMM_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    platforms/cuda/include
    openmmapi/include
    omm_headers
    kernel_src)
  if(APPLE)
    target_link_libraries(mcbarostat2CUDA
      ${OPENMM_LIBRARIES}
      ${OPENMM_PLUGIN_DIR}/libOpenMMCUDA.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcuda.dylib)
    set_property(TARGET mcbarostat2CUDA
      PROPERTY MACOSX_RPATH ON)
  endif()

  set(MCBAROSTAT2_TARGS ${MCBAROSTAT2_TARGS} mcbarostat2CUDA)
endif(CUDA_FOUND)

if(OPENCL_FOUND)
  add_library(mcbarostat2OpenCL SHARED
    kernel_src/OpenCLCharmmKernelSources.cpp
    platforms/opencl/src/OpenCLCharmmKernelFactory.cpp
    platforms/opencl/src/OpenCLCharmmKernels.cpp)
  target_include_directories(mcbarostat2OpenCL PRIVATE
    ${OPENMM_INCLUDE_DIRS}
    ${OPENCL_INCLUDE_DIR}
    openmmapi/include
    omm_headers
    platforms/opencl/include
    kernel_src)
  if(APPLE)
    target_link_libraries(mcbarostat2OpenCL
      ${OPENMM_LIBRARIES}
      ${OPENMM_PLUGIN_DIR}/libOpenMMOpenCL.dylib
      ${OPENCL_LIBRARY})
    set_property(TARGET mcbarostat2OpenCL
      PROPERTY MACOSX_RPATH ON)
  endif()
  set(MCBAROSTAT2_TARGS ${MCBAROSTAT2_TARGS} mcbarostat2OpenCL)
endif()

install(TARGETS ${MCBAROSTAT2_TARGS}
  DESTINATION ${CHARMM_PLUGIN_INSTALL_DIR})
