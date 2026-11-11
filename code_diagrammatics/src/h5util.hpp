// h5util.hpp -- thin HDF5 C++ helpers: complex {r,i} compound (h5py convention),
// attributes, whole-dataset read/write, and extendible (append-along-axis-0) datasets.
#pragma once

#include <H5Cpp.h>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace h5 {

using cplx = std::complex<double>;

H5::CompType complex_type();

// Open an existing HDF5 file read-only with clear errors (std::runtime_error, never H5::Exception):
// "<what> '<path>': does not exist | is not a regular file | is not readable | is not an HDF5 file | ...".
void check_readable_hdf5(const std::string& path, const std::string& what);
H5::H5File open_ro(const std::string& path, const std::string& what);
// Run f(); rethrow any H5::Exception or std::exception as std::runtime_error prefixed with `context`.
template <class F>
auto guarded(const std::string& context, F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const H5::Exception& e) {
    throw std::runtime_error(context + ": HDF5 error in " + e.getFuncName() + ": " + e.getDetailMsg());
  } catch (const std::exception& e) {
    throw std::runtime_error(context + ": " + e.what());
  }
}

bool has_attr(const H5::H5Object& o, const std::string& n);
bool attr_is_string(const H5::H5Object& o, const std::string& n);
void attr_str(H5::H5Object& o, const std::string& n, const std::string& v);
void attr_dbl(H5::H5Object& o, const std::string& n, double v);
void attr_int(H5::H5Object& o, const std::string& n, long long v);
void attr_dvec(H5::H5Object& o, const std::string& n, const std::vector<double>& v);
void attr_ivec(H5::H5Object& o, const std::string& n, const std::vector<long long>& v);
std::string read_attr_str(const H5::H5Object& o, const std::string& n);
double read_attr_dbl(const H5::H5Object& o, const std::string& n);
std::vector<double> read_attr_dvec(const H5::H5Object& o, const std::string& n);  // scalar -> length 1

// group helpers
bool exists(const H5::Group& g, const std::string& path);
H5::Group group(H5::Group parent, const std::string& name);  // open or create

// whole datasets
void write_cplx(H5::Group& g, const std::string& n, const cplx* d, const std::vector<hsize_t>& dims);
void write_dbl(H5::Group& g, const std::string& n, const double* d, const std::vector<hsize_t>& dims);
void write_i64(H5::Group& g, const std::string& n, const long long* d, const std::vector<hsize_t>& dims);
std::vector<cplx> read_cplx(const H5::Group& g, const std::string& n, std::vector<hsize_t>* dims = nullptr);
std::vector<double> read_dbl(const H5::Group& g, const std::string& n, std::vector<hsize_t>* dims = nullptr);
std::vector<long long> read_i64(const H5::Group& g, const std::string& n, std::vector<hsize_t>* dims = nullptr);

// dataset of shape [T, rest...], T unlimited, appended one row at a time
class Appender {
 public:
  enum Kind { Complex, Double };
  Appender() = default;
  Appender(H5::Group& g, const std::string& n, Kind k, const std::vector<hsize_t>& rest, hsize_t chunk_rows = 64);
  void append(const void* row);  // one row of prod(rest) elements
  hsize_t rows() const { return rows_; }
  bool valid() const { return valid_; }

 private:
  H5::DataSet ds_;
  Kind kind_ = Double;
  std::vector<hsize_t> rest_;
  hsize_t rows_ = 0;
  bool valid_ = false;
};

}  // namespace h5
