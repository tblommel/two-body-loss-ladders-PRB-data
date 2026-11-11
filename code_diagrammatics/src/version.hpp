// version.hpp -- build provenance (release version, git describe, compiler, flags). The strings are generated at
// every build by cmake/gen_version.cmake into <build>/generated/version_info.hpp and compiled into version.cpp only.
#pragma once

#include <string>

namespace drv {
std::string release_version();  // "<VERSION>+<git describe --always --dirty>" (or "<VERSION>+nogit")
std::string git_describe();      // git describe --always --dirty of the source tree, or "nogit"
std::string build_flags();       // build type and the full compile flags (incl. -march if any)
std::string compiler();          // compiler id, version and __VERSION__
std::string version_text();      // everything above plus Eigen/HDF5 versions, for dp5drive --version
}  // namespace drv
