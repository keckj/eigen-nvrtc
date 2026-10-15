// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NVRTC_SUPPORT_H
#define EIGEN_NVRTC_SUPPORT_H

// Support for NVRTC (CUDA runtime compilation).
//
// NVRTC compiles device code only and ships no C or C++ standard library:
// none of <cmath>, <limits>, <type_traits>, ... can be included. The only
// standard library available is libcu++ (CCCL), which provides a subset of the
// standard library in namespace cuda::std through <cuda/std/...> headers.
// In addition, NVRTC implicitly declares the CUDA device runtime functions in
// the global namespace (math functions such as ::sqrt or ::isnan, ::malloc,
// ::free, ::memcpy, ::memset, ::printf, ...).
//
// This header is included first by Eigen/Core. When compiling with NVRTC it:
//  - defines EIGEN_CUDACC_RTC, which is used to guard all system includes and
//    the few code paths that rely on host-only facilities,
//  - includes the libcu++ headers corresponding to the standard headers used
//    by Eigen,
//  - makes the libcu++ entities visible as std::xxx (see below),
//  - defines EIGEN_STD_NS_BEGIN / EIGEN_STD_NS_END that must be used instead of
//    "namespace std {" / "}" to specialize standard templates (numeric_limits).
//
// Requirements: the libcu++ headers must be reachable, i.e. NVRTC must be given
// "-I<cuda>/include" (or the CCCL include directory). C++17 or later is needed.
//
// When not compiling with NVRTC this header only defines EIGEN_STD_NS_BEGIN and
// EIGEN_STD_NS_END as "namespace std {" and "}".

#if defined(__CUDACC_RTC__)

#define EIGEN_CUDACC_RTC 1

#if __cplusplus < 201703L
#error "Eigen with NVRTC requires -std=c++17 or later."
#endif

// No system headers are available under NVRTC, these are the libcu++ replacements.
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/climits>
#include <cuda/std/cfloat>
#include <cuda/std/cstdlib>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/functional>
#include <cuda/std/initializer_list>
#include <cuda/std/iterator>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/complex>
#include <cuda/std/bit>
#include <cuda/std/memory>
#include <cuda/std/version>
#if __has_include(<cuda/std/algorithm>)
#include <cuda/std/algorithm>
#else
// CCCL 2.x only ships the algorithms through an internal header.
#include <cuda/std/__algorithm_>
#endif

// Make the libcu++ entities available as std::xxx.
//
// A using-directive is used rather than a list of using-declarations because:
//  - Eigen refers to well over a hundred different std entities (type traits,
//    std::numeric_limits, std::complex, math functions, functors, iterators,
//    std::array, ...) spread over all modules; listing them would be fragile
//    and any omission would be a hard error,
//  - user code compiled together with Eigen (e.g. automatic differentiation
//    libraries) usually relies on std:: names as well (std::is_same_v, ...),
//  - an explicit list would not help for explicit specializations anyway, see
//    EIGEN_STD_NS_BEGIN below.
// Qualified lookup of std::xxx searches the namespaces nominated by this
// directive, so std::xxx finds cuda::std::xxx. Declarations made directly in
// namespace std (below) take precedence over the nominated ones.
namespace std {
using namespace cuda::std;
}  // namespace std

// The C library functions used by Eigen that libcu++ does not provide but that
// the CUDA device runtime declares in the global namespace.
namespace std {
using ::free;
using ::malloc;
using ::memcpy;
using ::memset;
// <cmath> functions missing from libcu++ under NVRTC (CCCL 2.x), declared by
// the device runtime in the global namespace.
using ::cbrt;
using ::erf;
using ::erfc;
using ::exp2;
using ::expm1;
using ::fdim;
using ::floor;
using ::fma;
using ::fmod;
using ::frexp;
using ::ldexp;
using ::lgamma;
using ::llrint;
using ::llround;
using ::lrint;
using ::lround;
using ::modf;
using ::nan;
using ::nearbyint;
using ::nextafter;
using ::remainder;
using ::remquo;
using ::rint;
using ::round;
using ::scalbln;
using ::scalbn;
using ::tgamma;
using ::trunc;
// Tag type used by Eigen's nothrow operator new overloads. NVRTC has no <new>
// and therefore no std::nothrow_t; Eigen only needs the type to declare these
// overloads.
#if !defined(EIGEN_NVRTC_NO_NOTHROW_T)
struct nothrow_t {
  explicit nothrow_t() = default;
};
#endif
}  // namespace std

// Explicit specializations of standard templates must be declared in the
// namespace that actually declares the primary template, which is cuda::std
// under NVRTC (a using-directive or using-declaration is not enough).
#define EIGEN_STD_NS_BEGIN \
  namespace cuda {         \
  namespace std {
#define EIGEN_STD_NS_END \
  }                      \
  }

// There is no I/O under NVRTC.
#ifndef EIGEN_NO_IO
#define EIGEN_NO_IO
#endif

#else  // defined(__CUDACC_RTC__)

#define EIGEN_STD_NS_BEGIN namespace std {
#define EIGEN_STD_NS_END }

#endif  // defined(__CUDACC_RTC__)

#endif  // EIGEN_NVRTC_SUPPORT_H
