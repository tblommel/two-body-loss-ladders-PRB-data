cmake_minimum_required(VERSION 3.16)
project(dp5driver CXX)

# Build (README.md):
#   cmake -S code_diagrammatics -B code_diagrammatics/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build code_diagrammatics/build -j
# Portable by default (-O3 -DNDEBUG, no -march, no -ffast-math). Opt-in: -DDP5_NATIVE=ON (-march=native) or
# -DCMAKE_CXX_FLAGS=-march=x86-64-v3. Different flags change results only at the level of the tolerances.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
option(DP5_NATIVE "compile with -march=native (not portable: the binary only runs on CPUs like the build host)" OFF)
option(DP5_BUILD_TESTS "build the integrator unit test (integrator/tests/test_dp5)" ON)

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---- provenance: release version (../VERSION), git describe, compiler and flags, regenerated at every build
set(DP5_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION")
if(NOT EXISTS "${DP5_VERSION_FILE}")
  set(DP5_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/VERSION.missing")
  file(WRITE "${DP5_VERSION_FILE}" "unknown")
  message(WARNING "../VERSION not found: release version 'unknown'")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
set(_flags "${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}} ${OpenMP_CXX_FLAGS}")
if(DP5_NATIVE)
  set(_flags "${_flags} -march=native")
endif()
string(REGEX REPLACE "  +" " " _flags "${_flags}")
string(STRIP "${_flags}" _flags)
set(_gen "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_gen}")
add_custom_target(dp5_version ALL
  COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DVERSION_FILE=${DP5_VERSION_FILE}
          -DOUT=${_gen}/version_info.hpp "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
          "-DFLAGS=${_flags}" -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_version.cmake
  BYPRODUCTS ${_gen}/version_info.hpp
  COMMENT "dp5drive: version stamp"
  VERBATIM)

# ---- the driver
add_executable(dp5drive src/main.cpp src/inputs.cpp src/physics.cpp src/observables.cpp src/steady_state.cpp
                        src/output.cpp src/h5util.cpp src/version.cpp)
add_dependencies(dp5drive dp5_version)
target_include_directories(dp5drive PRIVATE ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/integrator ${_gen})
target_link_libraries(dp5drive PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen ${HDF5_LIBRARIES})
target_compile_definitions(dp5drive PRIVATE ${HDF5_DEFINITIONS})
if(DP5_NATIVE)
  target_compile_options(dp5drive PRIVATE -march=native)
endif()

# ---- integrator unit test (ctest / integrator/tests)
if(DP5_BUILD_TESTS)
  add_executable(test_dp5 integrator/tests/test_dp5.cpp)
  target_include_directories(test_dp5 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/integrator)
  target_link_libraries(test_dp5 PRIVATE OpenMP::OpenMP_CXX)
  if(DP5_NATIVE)
    target_compile_options(test_dp5 PRIVATE -march=native)
  endif()
  enable_testing()
  add_test(NAME integrator_test_dp5 COMMAND test_dp5)
endif()
