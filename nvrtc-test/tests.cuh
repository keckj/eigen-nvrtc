// Eigen NVRTC validation kernels.
//
// This file is compiled twice:
//  - by NVRTC (device code): every test function is wrapped in an
//    extern "C" __global__ kernel at the end of the file,
//  - by the host C++ compiler (reference results): the test functions are
//    called directly.
// Each test reads from `in` (deterministic input values) and writes
// out[0] = number of results, out[1..] = results (as double).
#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#if defined(__CUDACC_RTC__)
#define TEST_FUNC __host__ __device__
#else
#define TEST_FUNC
#endif

// Results writer.
struct Out {
  double* p;
  int n;
  TEST_FUNC explicit Out(double* p_) : p(p_), n(0) {}
  TEST_FUNC ~Out() { p[0] = n; }
  TEST_FUNC void put(double x) { p[1 + n++] = x; }
  template <typename Derived>
  TEST_FUNC void put(const Eigen::DenseBase<Derived>& m) {
    for (Eigen::Index j = 0; j < m.cols(); ++j)
      for (Eigen::Index i = 0; i < m.rows(); ++i) put(static_cast<double>(m(i, j)));
  }
};

// ---------------------------------------------------------------------------
// Custom scalar: a dual number (value + derivative), as used by forward mode
// automatic differentiation.
// ---------------------------------------------------------------------------
struct Dual {
  double v, d;
  TEST_FUNC Dual() : v(0), d(0) {}
  TEST_FUNC Dual(double v_) : v(v_), d(0) {}
  TEST_FUNC Dual(double v_, double d_) : v(v_), d(d_) {}
  TEST_FUNC Dual& operator+=(const Dual& o) { v += o.v; d += o.d; return *this; }
  TEST_FUNC Dual& operator-=(const Dual& o) { v -= o.v; d -= o.d; return *this; }
  TEST_FUNC Dual& operator*=(const Dual& o) { d = d * o.v + v * o.d; v *= o.v; return *this; }
  TEST_FUNC Dual& operator/=(const Dual& o) { d = (d * o.v - v * o.d) / (o.v * o.v); v /= o.v; return *this; }
};
TEST_FUNC inline Dual operator+(Dual a, const Dual& b) { return a += b; }
TEST_FUNC inline Dual operator-(Dual a, const Dual& b) { return a -= b; }
TEST_FUNC inline Dual operator*(Dual a, const Dual& b) { return a *= b; }
TEST_FUNC inline Dual operator/(Dual a, const Dual& b) { return a /= b; }
TEST_FUNC inline Dual operator-(const Dual& a) { return Dual(-a.v, -a.d); }
TEST_FUNC inline bool operator==(const Dual& a, const Dual& b) { return a.v == b.v; }
TEST_FUNC inline bool operator!=(const Dual& a, const Dual& b) { return a.v != b.v; }
TEST_FUNC inline bool operator<(const Dual& a, const Dual& b) { return a.v < b.v; }
TEST_FUNC inline bool operator>(const Dual& a, const Dual& b) { return a.v > b.v; }
TEST_FUNC inline bool operator<=(const Dual& a, const Dual& b) { return a.v <= b.v; }
TEST_FUNC inline bool operator>=(const Dual& a, const Dual& b) { return a.v >= b.v; }
TEST_FUNC inline Dual sqrt(const Dual& a) {
  using std::sqrt;
  double s = sqrt(a.v);
  return Dual(s, a.d / (2 * s));
}
TEST_FUNC inline Dual abs(const Dual& a) { return a.v < 0 ? -a : a; }
TEST_FUNC inline Dual conj(const Dual& a) { return a; }
TEST_FUNC inline Dual real(const Dual& a) { return a; }
TEST_FUNC inline Dual imag(const Dual&) { return Dual(0); }
TEST_FUNC inline Dual abs2(const Dual& a) { return a * a; }

namespace Eigen {
template <>
struct NumTraits<Dual> : NumTraits<double> {
  typedef Dual Real;
  typedef Dual NonInteger;
  typedef Dual Nested;
  enum { IsComplex = 0, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 2, AddCost = 2, MulCost = 4 };
};
}  // namespace Eigen

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_FUNC inline void test_arith(const double* in, double* o_) {
  Out o(o_);
  Eigen::Matrix3d A;
  A << in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8];
  Eigen::Vector3d v(in[9], in[10], in[11]);
  Eigen::Matrix3d B = A * A.transpose() + 2.0 * Eigen::Matrix3d::Identity();
  o.put(B);
  o.put(A * v);
  o.put((A - B) / 3.0);
  o.put(v.transpose() * A);
  Eigen::Matrix4f F = Eigen::Matrix4f::Constant(0.5f);
  F(1, 2) = float(in[3]);
  F.diagonal() << 1.f, 2.f, 3.f, 4.f;
  o.put(F * F);
  Eigen::Array<double, 2, 3> a, b;
  a << in[0], in[1], in[2], in[3], in[4], in[5];
  b << in[6], in[7], in[8], in[9], in[10], in[11];
  o.put(a * b + a / b - 1.0);
  o.put(a.max(b));
  o.put((a > b).cast<double>());
  Eigen::Matrix<int, 2, 2> I;
  I << 1, 2, 3, 4;
  o.put((I * I).cast<double>());
}

TEST_FUNC inline void test_cwise(const double* in, double* o_) {
  Out o(o_);
  Eigen::Array<double, 6, 1> x;
  for (int i = 0; i < 6; ++i) x[i] = in[i];
  o.put(x.sin());
  o.put(x.cos());
  o.put(x.exp());
  o.put(x.abs().sqrt());
  o.put(x.abs());
  o.put(x.abs().pow(1.5));
  o.put(x.abs().log());
  o.put(x.tanh());
  o.put(x.square());
  o.put(x.floor());
  o.put(x.round());
  o.put(Eigen::numext::isnan(x[0]) ? 1.0 : 0.0);
  double nanv = Eigen::NumTraits<double>::quiet_NaN();
  double infv = Eigen::NumTraits<double>::infinity();
  o.put((Eigen::numext::isnan)(nanv) ? 1.0 : 0.0);
  o.put((Eigen::numext::isinf)(infv) ? 1.0 : 0.0);
  o.put((Eigen::numext::isfinite)(infv) ? 1.0 : 0.0);
  o.put((Eigen::numext::isfinite)(x[1]) ? 1.0 : 0.0);
  Eigen::Array<double, 3, 1> y(nanv, 1.0, infv);
  o.put(y.isNaN().cast<double>());
  o.put(y.isInf().cast<double>());
  o.put(y.isFinite().cast<double>());
  Eigen::Array4f f{float(in[0]), float(in[1]), float(in[2]), float(in[3])};
  o.put(f.sin().cast<double>());
  o.put(f.exp().cast<double>());
  o.put(f.abs().sqrt().cast<double>());
}

TEST_FUNC inline void test_reduc(const double* in, double* o_) {
  Out o(o_);
  Eigen::Matrix<double, 4, 3> M;
  for (int i = 0; i < 12; ++i) M.data()[i] = in[i];
  Eigen::Vector3d a(in[0], in[1], in[2]), b(in[3], in[4], in[5]);
  o.put(M.sum());
  o.put(M.prod());
  o.put(M.mean());
  o.put(M.norm());
  o.put(M.squaredNorm());
  o.put(M.maxCoeff());
  o.put(M.minCoeff());
  Eigen::Index r, c;
  M.maxCoeff(&r, &c);
  o.put(double(r));
  o.put(double(c));
  o.put(M.colwise().sum());
  o.put(M.rowwise().norm());
  o.put(a.dot(b));
  o.put(a.normalized());
  o.put(a.cross(b));
  o.put(M.cwiseAbs().maxCoeff());
  o.put(M.lpNorm<1>());
  o.put(M.lpNorm<Eigen::Infinity>());
  o.put((M.array() > 0).count());
  o.put((M.array() > 0).any() ? 1.0 : 0.0);
  o.put(M.trace());
}

TEST_FUNC inline void test_block(const double* in, double* o_) {
  using namespace Eigen;
  Out o(o_);
  Matrix<double, 4, 4> M;
  for (int i = 0; i < 16; ++i) M.data()[i] = in[i];
  Matrix<double, 6, 1> v;
  for (int i = 0; i < 6; ++i) v[i] = in[16 + i];
  o.put(M.transpose());
  o.put(M.block<2, 3>(1, 0));
  o.put(M.block(0, 1, 3, 2));
  o.put(M.topLeftCorner<2, 2>());
  o.put(M.row(2));
  o.put(M.col(1));
  o.put(v.head<2>());
  o.put(v.tail(3));
  o.put(v.segment<3>(1));
  o.put(v.segment(2, 2));
  o.put(v(placeholders::last));
  o.put(v(seq(1, placeholders::last)));
  o.put(v(seq(0, placeholders::last, fix<2>)));
  o.put(v(seqN(fix<1>, fix<3>)));
  o.put(v(placeholders::lastN(2)));
  o.put(M(placeholders::all, seq(fix<1>, fix<2>)));
  o.put(M(seqN(0, 2), placeholders::last));
  o.put(v.reverse());
  Matrix4d N = M;
  N.block<2, 2>(2, 2).setZero();
  N.col(0).swap(N.col(3));
  o.put(N);
  o.put(M.diagonal());
  Matrix<double, 2, 8> R = M.reshaped<RowMajor>(2, 8);
  o.put(R);
}

TEST_FUNC inline void test_map(const double* in, double* o_) {
  Out o(o_);
  Eigen::Map<const Eigen::Matrix3d> A(in);
  Eigen::Map<const Eigen::Vector3d> v(in + 9);
  o.put(A * v);
  Eigen::Map<const Eigen::Matrix<double, 3, 2>, 0, Eigen::OuterStride<4>> S(in);
  o.put(S);
  Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<2>> s(in, 5);
  o.put(s);
  double buf[6];
  Eigen::Map<Eigen::Matrix<double, 2, 3>> W(buf);
  W.setConstant(in[2]);
  W(1, 1) = in[3];
  o.put(W);
  // Dynamic size: heap allocation on the device (malloc/free).
  Eigen::VectorXd d(7);
  for (int i = 0; i < 7; ++i) d[i] = in[i];
  Eigen::MatrixXd D = d * d.transpose();
  o.put(D.sum());
  o.put(D.diagonal());
  d.conservativeResize(9);
  d.tail<2>().setOnes();
  o.put(d);
}

TEST_FUNC inline void test_lu(const double* in, double* o_) {
  Out o(o_);
  Eigen::Matrix3d A;
  for (int i = 0; i < 9; ++i) A.data()[i] = in[i];
  A += 3 * Eigen::Matrix3d::Identity();
  Eigen::Matrix4d B;
  for (int i = 0; i < 16; ++i) B.data()[i] = in[i + 3];
  B += 4 * Eigen::Matrix4d::Identity();
  Eigen::Vector3d b3(in[0], in[1], in[2]);
  Eigen::Vector4d b4(in[3], in[4], in[5], in[6]);
  o.put(A.determinant());
  o.put(B.determinant());
  o.put(A.inverse());
  o.put(B.inverse());
  Eigen::Matrix2d C = A.topLeftCorner<2, 2>();
  o.put(C.inverse());
  o.put(A.partialPivLu().solve(b3));
  o.put(B.partialPivLu().solve(b4));
  Eigen::PartialPivLU<Eigen::Matrix4d> lu(B);
  o.put(lu.determinant());
  o.put(B.fullPivLu().solve(b4));
  o.put(double(A.fullPivLu().rank()));
  Eigen::Matrix4f Bf = B.cast<float>();
  o.put(Bf.inverse().cast<double>());
}

TEST_FUNC inline void test_chol(const double* in, double* o_) {
  Out o(o_);
  Eigen::Matrix3d A;
  for (int i = 0; i < 9; ++i) A.data()[i] = in[i];
  Eigen::Matrix3d S = A * A.transpose() + Eigen::Matrix3d::Identity();
  Eigen::Vector3d b(in[9], in[10], in[11]);
  Eigen::LLT<Eigen::Matrix3d> llt(S);
  o.put(llt.solve(b));
  o.put(Eigen::Matrix3d(llt.matrixL()));
  Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
  o.put(ldlt.solve(b));
  o.put(ldlt.vectorD());
  Eigen::Matrix4d A4;
  for (int i = 0; i < 16; ++i) A4.data()[i] = in[i];
  Eigen::Matrix4d S4 = A4.transpose() * A4 + Eigen::Matrix4d::Identity();
  o.put(S4.llt().solve(Eigen::Vector4d::Ones()));
  o.put(S4.ldlt().solve(Eigen::Vector4d::Ones()));
  o.put(S4.selfadjointView<Eigen::Lower>() * Eigen::Vector4d::Ones());
  o.put(Eigen::Matrix4d(S4.triangularView<Eigen::Upper>()));
  o.put(S4.triangularView<Eigen::Lower>().solve(Eigen::Vector4d::Ones()));
}

TEST_FUNC inline void test_qr(const double* in, double* o_) {
  Out o(o_);
  Eigen::Matrix<double, 4, 3> A;
  for (int i = 0; i < 12; ++i) A.data()[i] = in[i];
  Eigen::Vector4d b(in[12], in[13], in[14], in[15]);
  Eigen::HouseholderQR<Eigen::Matrix<double, 4, 3>> qr(A);
  o.put(qr.solve(b));
  o.put(qr.matrixQR().cwiseAbs());
  Eigen::Matrix3d S;
  for (int i = 0; i < 9; ++i) S.data()[i] = in[i + 2];
  Eigen::Vector3d c(in[0], in[1], in[2]);
  o.put(S.householderQr().solve(c));
  o.put(S.colPivHouseholderQr().solve(c));
  o.put(double(A.colPivHouseholderQr().rank()));
}

TEST_FUNC inline void test_eig(const double* in, double* o_) {
  Out o(o_);
  Eigen::Matrix3d A;
  for (int i = 0; i < 9; ++i) A.data()[i] = in[i];
  Eigen::Matrix3d S = A + A.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(S);
  o.put(es.eigenvalues());
  o.put(es.eigenvectors().cwiseAbs());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> ed;
  ed.computeDirect(S);
  o.put(ed.eigenvalues());
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  o.put(svd.singularValues());
  o.put(svd.matrixU().cwiseAbs());
  o.put(svd.solve(Eigen::Vector3d(in[9], in[10], in[11])));
  Eigen::Matrix<double, 4, 3> A43 = Eigen::Map<const Eigen::Matrix<double, 4, 3>>(in);
  Eigen::JacobiSVD<Eigen::Matrix<double, 4, 3>> svd2(A43);
  o.put(svd2.singularValues());
}

TEST_FUNC inline void test_geo(const double* in, double* o_) {
  Out o(o_);
  Eigen::Vector3d axis = Eigen::Vector3d(in[0], in[1], in[2]).normalized();
  Eigen::AngleAxisd aa(in[3], axis);
  Eigen::Quaterniond q(aa);
  Eigen::Vector3d v(in[4], in[5], in[6]);
  o.put(q.coeffs());
  o.put(q * v);
  o.put(q.toRotationMatrix());
  Eigen::Quaterniond p(Eigen::AngleAxisd(in[7], Eigen::Vector3d::UnitZ()));
  o.put((p * q).coeffs());
  o.put(q.slerp(0.3, p).coeffs());
  o.put(q.inverse().coeffs());
  o.put(q.angularDistance(p));
  Eigen::AngleAxisd back(q);
  o.put(back.angle());
  o.put(back.axis());
  Eigen::Quaterniond f = Eigen::Quaterniond::FromTwoVectors(v, axis);
  o.put(f.coeffs());
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(q);
  T.pretranslate(Eigen::Vector3d(in[8], in[9], in[10]));
  o.put(T * v);
  o.put(T.inverse().matrix());
  Eigen::Matrix3d R = q.toRotationMatrix();
  o.put(R.canonicalEulerAngles(2, 0, 2).cwiseAbs());
  Eigen::Rotation2Dd r2(in[3]);
  o.put(r2.toRotationMatrix());
  Eigen::Vector3d h = (T.matrix() * v.homogeneous()).hnormalized();
  o.put(h);
}

TEST_FUNC inline void test_half(const double* in, double* o_) {
  Out o(o_);
  using Eigen::half;
  using Eigen::bfloat16;
  Eigen::Matrix<half, 3, 3> H;
  Eigen::Matrix<bfloat16, 3, 3> B;
  for (int i = 0; i < 9; ++i) {
    H.data()[i] = half(float(in[i]));
    B.data()[i] = bfloat16(float(in[i]));
  }
  Eigen::Matrix<half, 3, 3> H2 = H * H + H;
  Eigen::Matrix<bfloat16, 3, 3> B2 = B * B - B;
  for (int i = 0; i < 9; ++i) o.put(double(float(H2.data()[i])));
  for (int i = 0; i < 9; ++i) o.put(double(float(B2.data()[i])));
  o.put(double(float(H.sum())));
  o.put(double(float(B.maxCoeff())));
  half a = half(1.5f), b = half(float(in[2]));
  o.put(double(float(a * b + a / b)));
  o.put(double(float(Eigen::numext::sqrt(a))));
  o.put(double(float(Eigen::numext::exp(b))));
  bfloat16 c = bfloat16(2.25f), d = bfloat16(float(in[3]));
  o.put(double(float(c * d - c)));
  o.put(double(float(Eigen::numext::abs(d))));
  o.put(double(float(Eigen::NumTraits<half>::epsilon())));
  o.put(double(float(Eigen::NumTraits<bfloat16>::epsilon())));
  o.put(double(float((std::numeric_limits<half>::max)())));
  o.put(double(float((std::numeric_limits<bfloat16>::max)())));
}

TEST_FUNC inline void test_custom(const double* in, double* o_) {
  Out o(o_);
  Eigen::Matrix<Dual, 3, 3> A, B;
  for (int i = 0; i < 9; ++i) {
    A.data()[i] = Dual(in[i], 1.0);
    B.data()[i] = Dual(in[i + 9], 0.5);
  }
  Eigen::Matrix<Dual, 3, 3> C = A * B;
  for (int i = 0; i < 9; ++i) {
    o.put(C.data()[i].v);
    o.put(C.data()[i].d);
  }
  Eigen::Matrix<Dual, 3, 1> v = A.col(0);
  Dual n = v.norm();
  o.put(n.v);
  o.put(n.d);
  Dual s = (A + B).sum();
  o.put(s.v);
  o.put(s.d);
  Dual dt = A.row(1).dot(B.row(2));
  o.put(dt.v);
  o.put(dt.d);
  Eigen::Matrix<Dual, 3, 1> w = (A.transpose() * v) / Dual(2.0);
  for (int i = 0; i < 3; ++i) {
    o.put(w[i].v);
    o.put(w[i].d);
  }
}

#define EIGEN_NVRTC_TESTS(X) \
  X(arith)                   \
  X(cwise)                   \
  X(reduc)                   \
  X(block)                   \
  X(map)                     \
  X(lu)                      \
  X(chol)                    \
  X(qr)                      \
  X(eig)                     \
  X(geo)                     \
  X(half)                    \
  X(custom)

#if defined(__CUDACC_RTC__)
#define EIGEN_NVRTC_KERNEL(name) \
  extern "C" __global__ void kernel_##name(const double* in, double* out) { test_##name(in, out); }
EIGEN_NVRTC_TESTS(EIGEN_NVRTC_KERNEL)
#endif
