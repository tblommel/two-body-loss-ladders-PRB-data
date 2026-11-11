// common.hpp -- shared types and switches of the dp5 driver.
#pragma once

#include <Eigen/Dense>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "dp5_integrator.hpp"  // vendored in ../integrator (include path set by CMake)

namespace drv {

using cplx = std::complex<double>;
using Mat = Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatMap = Eigen::Map<Mat>;
using CMatMap = Eigen::Map<const Mat>;
using Vec = Eigen::Matrix<cplx, Eigen::Dynamic, 1>;

// Schema versions of the three file types (README, "File schemas").
constexpr int kModelSchema = 1;
constexpr int kRunCardSchema = 1;
constexpr int kOutputSchema = 1;

enum class Approx { HF, TPPX, mGW, DSL };
enum class Mode { g1g2, rtde };
enum class SpinMode { full };  // the one-spin (paramagnetic) layout of the development driver is not part of the release
enum class VertexImpl { general, impurity };

inline Approx parse_approx(const std::string& s) {
  if (s == "HF") return Approx::HF;
  if (s == "TPPX") return Approx::TPPX;
  if (s == "BARE")  // removed 2026-10-28: BARE = TPPX with the bare-propagator flag (TPPX_0)
    throw std::invalid_argument("approximation 'BARE' was removed: use approximation = TPPX with bare_propagators = 1 "
                                "(method TPPX_0, identical to the old BARE)");
  if (s == "mGW") return Approx::mGW;
  if (s == "DSL") return Approx::DSL;
  throw std::invalid_argument("unknown approximation '" + s + "' (HF, TPPX, mGW, DSL)");
}
inline const char* name(Approx a) {
  switch (a) {
    case Approx::HF: return "HF";
    case Approx::TPPX: return "TPPX";
    case Approx::mGW: return "mGW";
    case Approx::DSL: return "DSL";
  }
  return "?";
}
// Reporting label: the approximation (TPPX + flag_2B = "2B"), with suffix "_0" when the correlated objects are
// propagated with the bare h_o (run card bare_propagators = 1). TPPX_0 is the former approximation BARE.
inline std::string method_label(Approx a, bool flag_2B, bool bare_propagators) {
  std::string s = (a == Approx::TPPX && flag_2B) ? std::string("2B") : std::string(name(a));
  return bare_propagators ? s + "_0" : s;
}
inline Mode parse_mode(const std::string& s) {
  if (s == "g1g2") return Mode::g1g2;
  if (s == "rtde") return Mode::rtde;
  throw std::invalid_argument("unknown mode '" + s + "' (g1g2, rtde)");
}
inline const char* name(Mode m) {
  switch (m) {
    case Mode::g1g2: return "g1g2";
    case Mode::rtde: return "rtde";
  }
  return "?";
}
inline SpinMode parse_spin(const std::string& s) {
  if (s == "full") return SpinMode::full;
  throw std::invalid_argument("unknown spin_mode '" + s + "' (only full)");
}
inline const char* name(SpinMode) { return "full"; }
inline VertexImpl parse_vertex(const std::string& s) {
  if (s == "general") return VertexImpl::general;
  if (s == "impurity") return VertexImpl::impurity;
  throw std::invalid_argument("unknown vertex_impl '" + s + "' (general, impurity)");
}
inline const char* name(VertexImpl v) { return v == VertexImpl::general ? "general" : "impurity"; }

struct NotImplemented : std::runtime_error {
  explicit NotImplemented(const std::string& what) : std::runtime_error("NOT IMPLEMENTED: " + what) {}
};

}  // namespace drv
