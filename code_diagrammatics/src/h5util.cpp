#include "h5util.hpp"

#include <unistd.h>

#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace h5 {

H5::CompType complex_type() {
  H5::CompType t(sizeof(cplx));
  t.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
  t.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
  return t;
}

void check_readable_hdf5(const std::string& path, const std::string& what) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (path.empty()) throw std::runtime_error(what + ": no path given");
  if (!fs::exists(path, ec)) throw std::runtime_error(what + " '" + path + "' does not exist");
  if (!fs::is_regular_file(path, ec)) throw std::runtime_error(what + " '" + path + "' is not a regular file");
  if (::access(path.c_str(), R_OK) != 0) throw std::runtime_error(what + " '" + path + "' is not readable (permission denied)");
  htri_t ok = H5Fis_hdf5(path.c_str());
  if (ok <= 0) throw std::runtime_error(what + " '" + path + "' is not an HDF5 file");
}
H5::H5File open_ro(const std::string& path, const std::string& what) {
  check_readable_hdf5(path, what);
  return guarded(what + " '" + path + "'", [&] { return H5::H5File(path, H5F_ACC_RDONLY); });
}

bool has_attr(const H5::H5Object& o, const std::string& n) { return o.attrExists(n); }
bool attr_is_string(const H5::H5Object& o, const std::string& n) {
  return o.openAttribute(n).getTypeClass() == H5T_STRING;
}

static void drop(H5::H5Object& o, const std::string& n) {
  if (o.attrExists(n)) o.removeAttr(n);
}
void attr_str(H5::H5Object& o, const std::string& n, const std::string& v) {
  drop(o, n);
  H5::StrType t(H5::PredType::C_S1, H5T_VARIABLE);
  t.setCset(H5T_CSET_UTF8);
  H5::Attribute a = o.createAttribute(n, t, H5::DataSpace(H5S_SCALAR));
  a.write(t, v);
}
void attr_dbl(H5::H5Object& o, const std::string& n, double v) {
  drop(o, n);
  H5::Attribute a = o.createAttribute(n, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(H5S_SCALAR));
  a.write(H5::PredType::NATIVE_DOUBLE, &v);
}
void attr_int(H5::H5Object& o, const std::string& n, long long v) {
  drop(o, n);
  H5::Attribute a = o.createAttribute(n, H5::PredType::NATIVE_LLONG, H5::DataSpace(H5S_SCALAR));
  a.write(H5::PredType::NATIVE_LLONG, &v);
}
void attr_dvec(H5::H5Object& o, const std::string& n, const std::vector<double>& v) {
  drop(o, n);
  hsize_t d = v.size();
  H5::DataSpace sp = d ? H5::DataSpace(1, &d) : H5::DataSpace(H5S_NULL);
  H5::Attribute a = o.createAttribute(n, H5::PredType::NATIVE_DOUBLE, sp);
  if (d) a.write(H5::PredType::NATIVE_DOUBLE, v.data());
}
void attr_ivec(H5::H5Object& o, const std::string& n, const std::vector<long long>& v) {
  drop(o, n);
  hsize_t d = v.size();
  H5::DataSpace sp = d ? H5::DataSpace(1, &d) : H5::DataSpace(H5S_NULL);
  H5::Attribute a = o.createAttribute(n, H5::PredType::NATIVE_LLONG, sp);
  if (d) a.write(H5::PredType::NATIVE_LLONG, v.data());
}
std::string read_attr_str(const H5::H5Object& o, const std::string& n) {
  if (!o.attrExists(n)) throw std::runtime_error("missing attribute '" + n + "'");
  H5::Attribute a = o.openAttribute(n);
  std::string s;
  a.read(a.getStrType(), s);
  return s;
}
std::vector<double> read_attr_dvec(const H5::H5Object& o, const std::string& n) {
  if (!o.attrExists(n)) throw std::runtime_error("missing attribute '" + n + "'");
  H5::Attribute a = o.openAttribute(n);
  H5::DataSpace sp = a.getSpace();
  if (sp.getSimpleExtentType() == H5S_NULL) return {};
  const hssize_t np = sp.getSimpleExtentNpoints();
  std::vector<double> v(static_cast<std::size_t>(np));
  if (np) a.read(H5::PredType::NATIVE_DOUBLE, v.data());
  return v;
}
double read_attr_dbl(const H5::H5Object& o, const std::string& n) {
  auto v = read_attr_dvec(o, n);
  if (v.size() != 1) throw std::runtime_error("attribute '" + n + "' is not a scalar");
  return v[0];
}

bool exists(const H5::Group& g, const std::string& path) {
  return H5Lexists(g.getId(), path.c_str(), H5P_DEFAULT) > 0;
}
H5::Group group(H5::Group parent, const std::string& name) {
  return exists(parent, name) ? parent.openGroup(name) : parent.createGroup(name);
}

static H5::DataSpace space(const std::vector<hsize_t>& dims) {
  return H5::DataSpace(int(dims.size()), dims.data());
}
void write_cplx(H5::Group& g, const std::string& n, const cplx* d, const std::vector<hsize_t>& dims) {
  if (exists(g, n)) g.unlink(n);
  auto t = complex_type();
  H5::DataSet ds = g.createDataSet(n, t, space(dims));
  if (space(dims).getSimpleExtentNpoints() > 0) ds.write(d, t);
}
void write_dbl(H5::Group& g, const std::string& n, const double* d, const std::vector<hsize_t>& dims) {
  if (exists(g, n)) g.unlink(n);
  H5::DataSet ds = g.createDataSet(n, H5::PredType::NATIVE_DOUBLE, space(dims));
  if (space(dims).getSimpleExtentNpoints() > 0) ds.write(d, H5::PredType::NATIVE_DOUBLE);
}
void write_i64(H5::Group& g, const std::string& n, const long long* d, const std::vector<hsize_t>& dims) {
  if (exists(g, n)) g.unlink(n);
  H5::DataSet ds = g.createDataSet(n, H5::PredType::NATIVE_LLONG, space(dims));
  if (space(dims).getSimpleExtentNpoints() > 0) ds.write(d, H5::PredType::NATIVE_LLONG);
}
template <class T>
static std::vector<T> read_any(const H5::Group& g, const std::string& n, const H5::DataType& t,
                               std::vector<hsize_t>* dims) {
  if (!exists(g, n)) throw std::runtime_error("missing dataset '" + n + "'");
  H5::DataSet ds = g.openDataSet(n);
  H5::DataSpace sp = ds.getSpace();
  const int r = sp.getSimpleExtentNdims();
  std::vector<hsize_t> d(static_cast<std::size_t>(r));
  if (r) sp.getSimpleExtentDims(d.data());
  std::vector<T> v(static_cast<std::size_t>(sp.getSimpleExtentNpoints()));
  if (!v.empty()) ds.read(v.data(), t);
  if (dims) *dims = d;
  return v;
}
std::vector<cplx> read_cplx(const H5::Group& g, const std::string& n, std::vector<hsize_t>* dims) {
  return read_any<cplx>(g, n, complex_type(), dims);
}
std::vector<double> read_dbl(const H5::Group& g, const std::string& n, std::vector<hsize_t>* dims) {
  return read_any<double>(g, n, H5::PredType::NATIVE_DOUBLE, dims);
}
std::vector<long long> read_i64(const H5::Group& g, const std::string& n, std::vector<hsize_t>* dims) {
  return read_any<long long>(g, n, H5::PredType::NATIVE_LLONG, dims);
}

Appender::Appender(H5::Group& g, const std::string& n, Kind k, const std::vector<hsize_t>& rest, hsize_t chunk_rows)
    : kind_(k), rest_(rest) {
  std::vector<hsize_t> dims{0}, maxd{H5S_UNLIMITED}, chunk{chunk_rows};
  for (auto r : rest) { dims.push_back(r); maxd.push_back(r); chunk.push_back(r); }
  H5::DataSpace sp(int(dims.size()), dims.data(), maxd.data());
  H5::DSetCreatPropList pl;
  pl.setChunk(int(chunk.size()), chunk.data());
  if (exists(g, n)) g.unlink(n);
  if (kind_ == Complex) ds_ = g.createDataSet(n, complex_type(), sp, pl);
  else ds_ = g.createDataSet(n, H5::PredType::NATIVE_DOUBLE, sp, pl);
  valid_ = true;
}
void Appender::append(const void* row) {
  std::vector<hsize_t> dims{rows_ + 1}, start{rows_}, count{1};
  for (auto r : rest_) { dims.push_back(r); start.push_back(0); count.push_back(r); }
  ds_.extend(dims.data());
  H5::DataSpace fs = ds_.getSpace();
  fs.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
  H5::DataSpace ms(int(count.size()), count.data());
  if (kind_ == Complex) ds_.write(row, complex_type(), ms, fs);
  else ds_.write(row, H5::PredType::NATIVE_DOUBLE, ms, fs);
  ++rows_;
}

}  // namespace h5
