#include "version.hpp"

#include <H5Cpp.h>

#include <Eigen/Core>

#include "version_info.hpp"  // generated at build time (DP5_RELEASE_VERSION, DP5_GIT_DESCRIBE, DP5_BUILD_FLAGS, DP5_COMPILER)

namespace drv {
std::string release_version() { return DP5_RELEASE_VERSION; }
std::string git_describe() { return DP5_GIT_DESCRIBE; }
std::string build_flags() { return DP5_BUILD_FLAGS; }
std::string compiler() {
#ifdef __VERSION__
  return std::string(DP5_COMPILER) + " (" + __VERSION__ + ")";
#else
  return DP5_COMPILER;
#endif
}
std::string version_text() {
  unsigned maj, min, rel;
  H5get_libversion(&maj, &min, &rel);
  return "dp5drive " + release_version() + "\n  git_describe  " + git_describe() + "\n  compiler      " + compiler() +
         "\n  build_flags   " + build_flags() + "\n  eigen_version " + std::to_string(EIGEN_WORLD_VERSION) + "." +
         std::to_string(EIGEN_MAJOR_VERSION) + "." + std::to_string(EIGEN_MINOR_VERSION) + "\n  hdf5_version  " +
         std::to_string(maj) + "." + std::to_string(min) + "." + std::to_string(rel) + "\n";
}
}  // namespace drv
