# Using Eigen with NVRTC

This branch (`nvrtc-5.0.1`, based on the Eigen 5.0.1 tag) makes Eigen usable
from device code compiled at runtime with NVRTC, the CUDA runtime compilation
library. All changes are guarded by `EIGEN_CUDACC_RTC`, which is only defined
when `__CUDACC_RTC__` is defined (i.e. by NVRTC). Builds with g++, clang, MSVC
or nvcc see exactly the same token stream as with pristine Eigen 5.0.1 (checked
by comparing the preprocessed output of `Eigen/Dense`, `Eigen/Sparse` and
`Eigen/Geometry` with g++ (C++17 and C++20) and with both nvcc phases: only
`__LINE__` values in assertion messages and blank lines differ).

## Requirements and options

- CUDA 12.x NVRTC with the libcu++ (CCCL) headers. Tested with CUDA 12.9
  (CCCL 2.8.2) on sm_120.
- C++17 or later (`-std=c++17` or `-std=c++20`). An `#error` is raised
  otherwise.
- `-default-device` is REQUIRED (see below).
- Include paths: the Eigen root and the CUDA include directory (for
  `<cuda/std/...>` and `<cuda_fp16.h>`).

Typical option set:

    -std=c++17 -default-device --gpu-architecture=compute_120
    -I/path/to/eigen -I/usr/local/cuda/include

Nothing else is needed: no `-D` macro, no pre-include, no
`--pre-include`. `-expt-relaxed-constexpr` does not exist in NVRTC and is not
needed.

### Why `-default-device` is required

Without `-default-device`, NVRTC rejects every function declaration that has
no execution space annotation, including uninstantiated templates,
declaration-only member templates and constexpr functions ("A function without
execution space annotations (__host__/__device__/__global__) is considered a
host function, and host functions are not allowed in JIT mode"). This is a
hard error that cannot be suppressed or turned into a warning, and there is
no pragma equivalent to `-default-device`.

Eigen leaves a large number of functions unannotated (type-trait helpers,
symbolic index operators, IntegralConstant, Memory.h, the CPU product kernels,
...). An automated experiment that added `__host__ __device__` to every
function NVRTC complained about needed more than 1000 annotations in 72
files just to compile a `Matrix3d * Vector3d` kernel, and still had not
converged (some declarations cannot be annotated mechanically). Such a patch
would not be upstreamable, so this port relies on `-default-device`, which
makes NVRTC treat unannotated functions and namespace-scope variables as
`__device__`. Without the flag the compilation fails at the first such
declaration (Meta.h).

## Usage

    #include <Eigen/Dense>      // or Eigen/Core, Eigen/Geometry, ...

    extern "C" __global__ void kernel(const double* in, double* out) {
      Eigen::Map<const Eigen::Matrix3d> A(in);
      Eigen::Vector3d b(in[9], in[10], in[11]);
      Eigen::Map<Eigen::Vector3d> x(out);
      x = A.partialPivLu().solve(b);
    }

Include Eigen before any header that uses `std::` names (see "std namespace"
below). Do not include standard headers (`<cmath>`, `<type_traits>`, ...)
yourself under NVRTC: they do not exist. Guard them with
`#ifndef __CUDACC_RTC__` (or `EIGEN_CUDACC_RTC` after Eigen is included).

## What is supported

Validated by `nvrtc-test/` (GPU results compared with the same code compiled
for the host by g++), with `-std=c++17` and `-std=c++20`:

- fixed-size Matrix/Vector/Array arithmetic (double, float, int), products,
  coefficient-wise operations and comparisons;
- coefficient-wise math: sin, cos, exp, log, sqrt, abs, pow, tanh, square,
  floor, round; isnan/isinf/isfinite (CUDA device builtins `::isnan` etc.);
- reductions: sum, prod, mean, norm, squaredNorm, lpNorm, min/maxCoeff (with
  index), dot, cross, colwise/rowwise, count, any, trace;
- transpose, block, head, tail, segment (compile-time and runtime sizes),
  corners, row/col, diagonal, reverse, swap, reshaped, indexed views with
  `Eigen::seq`, `seqN`, `Eigen::placeholders::last`, `lastN`, `all`,
  `Eigen::fix<N>`;
- Map over device pointers, with inner/outer strides;
- dynamic-size objects (VectorXd, MatrixXd, conservativeResize): allocation
  goes through the device heap (`malloc`/`free`);
- LU: determinant, inverse (2x2, 3x3, 4x4, double and float), PartialPivLU,
  FullPivLU (solve, rank);
- LLT, LDLT, triangular and self-adjoint views, triangular solve;
- HouseholderQR, ColPivHouseholderQR;
- JacobiSVD (3x3, 4x3), SelfAdjointEigenSolver (iterative and computeDirect);
- Geometry: Quaternion, AngleAxis, slerp, FromTwoVectors, Isometry3d,
  Rotation2D, canonicalEulerAngles, homogeneous/hnormalized;
- `Eigen::half` and `Eigen::bfloat16` matrices and scalar operations,
  `std::numeric_limits<half>`;
- custom scalar types (a dual number type with its own NumTraits, sqrt, abs)
  in matrix products, norm, dot, sum.

## Limitations

- `-default-device` is required (see above).
- Random number generation (`Random()`, `setRandom()`, `internal::random`):
  there is no `std::rand` on the device; it compiles but calls `__trap()` at
  runtime.
- No I/O: `EIGEN_NO_IO` is defined (no `operator<<`, no `IOFormat`).
- `std::string` is not available: `NumTraits<std::string>` and the
  `debugName()` members of the evaluator shape tags are not defined.
- `setCpuCacheSizes()` is ignored and the default cache sizes are used (only
  relevant for large dynamic-size products, which should be avoided in device
  code anyway).
- Large dynamic-size objects use the device heap (default 8 MB, see
  `cudaLimitMallocHeapSize`).
- `std::hash` specializations for half/bfloat16 are not provided (as with
  nvcc device code).
- Compilation is slow when many decompositions are instantiated in one
  program (about 160 s for the full test file, which instantiates all of the
  above); a single small kernel takes 2 to 4 seconds.
- With a global `using namespace std;`, unqualified calls such as `abs(1.0)`
  or `isnan(x)` are ambiguous between the NVRTC builtins and libcu++. This is
  a property of libcu++ under NVRTC (the same happens with
  `using namespace cuda::std;` and no Eigen). Qualify the call (`std::abs`) or
  use a block-scope using-declaration (`using std::abs;`).

## Design of the shim (Eigen/src/Core/util/NVRTCSupport.h)

`Eigen/Core` includes this header right after `Eigen/Version`, before anything
else. Under NVRTC it:

1. defines `EIGEN_CUDACC_RTC` (and raises an error before C++17);
2. includes the libcu++ headers corresponding to the standard headers Eigen
   uses (`<cuda/std/cmath>`, `<cuda/std/limits>`, `<cuda/std/type_traits>`,
   `<cuda/std/complex>`, `<cuda/std/array>`, `<cuda/std/bit>`, ...). CCCL 2.x
   does not provide a public `<cuda/std/algorithm>`; the internal
   `<cuda/std/__algorithm_>` is used in that case;
3. adds `namespace std { using namespace cuda::std; }`, so that every
   `std::xxx` used by Eigen (and by user code, e.g. `std::is_same_v`,
   `std::enable_if_t`, `std::numeric_limits`, `std::sqrt`) resolves to the
   libcu++ entity. A using-directive is used instead of a list of
   using-declarations because Eigen uses well over a hundred std names and a
   missing one is a hard error;
4. adds using-declarations in namespace std for the C library functions that
   libcu++ does not provide under NVRTC but that the CUDA device runtime
   declares globally: `malloc`, `free`, `memcpy`, `memset`, and the math
   functions `cbrt erf erfc exp2 expm1 fdim floor fma fmod frexp ldexp lgamma
   llrint llround lrint lround modf nan nearbyint nextafter remainder remquo
   rint round scalbln scalbn tgamma trunc`. Direct declarations in std take
   precedence over the nominated cuda::std ones, so this stays unambiguous if
   a later CCCL adds them;
5. declares `std::nothrow_t` (there is no `<new>`), needed by Eigen's
   nothrow operator new overloads. Define `EIGEN_NVRTC_NO_NOTHROW_T` if
   another header already provides it;
6. defines `EIGEN_STD_NS_BEGIN` / `EIGEN_STD_NS_END` as
   `namespace cuda { namespace std {` / `} }`: explicit specializations
   (`numeric_limits<half>`) must be declared in the namespace of the primary
   template, which is cuda::std. Outside NVRTC they expand to
   `namespace std {` / `}`;
7. defines `EIGEN_NO_IO`.

Outside NVRTC the header only defines `EIGEN_STD_NS_BEGIN` and
`EIGEN_STD_NS_END`.

Under NVRTC Eigen behaves like nvcc device code: `EIGEN_COMP_NVCC`,
`EIGEN_CUDACC`, `EIGEN_CUDA_ARCH`, `EIGEN_GPU_COMPILE_PHASE` are defined, so
the existing GPU code paths are used (math functions from the global
namespace, no CPU SIMD, `cuda_fp16.h` for half).

## Patch list

| File | Change | Rationale |
|------|--------|-----------|
| Eigen/src/Core/util/NVRTCSupport.h | new | shim described above |
| Eigen/Core | include NVRTCSupport.h first; skip `<cuda_runtime.h>`, `<new>`, `<complex>` and the block of system includes (`<cerrno>` ... `<intrin.h>`) | no system headers under NVRTC; libcu++ replacements come from the shim; the runtime functions are implicitly declared |
| Eigen/Geometry | skip `<limits>` | same |
| util/Macros.h | `EIGEN_COMP_NVCC` also for `__CUDACC_RTC__`; `EIGEN_CUDA_SDK_VER` from `__CUDACC_VER_MAJOR__/MINOR__` instead of `<cuda.h>`; skip `<cmath>`; `EIGEN_USING_STD` uses the global namespace on the device for NVRTC as for nvcc | NVRTC shares the nvcc front-end; `<cuda.h>` is not available; device math functions are global builtins |
| util/ConfigureVectorization.h | skip `<vector_types.h>` and `<cuda_runtime_api.h>`; hide `SimdInstructionSetsInUse()` | headers not available (types are builtin); host-only function |
| util/Meta.h, util/Serializer.h | skip `<cfloat>`, `<cstdint>`, `<type_traits>` | provided by the shim |
| util/Assert.h | `<cuda/std/cassert>` instead of `<cassert>` | no `<cassert>` |
| util/Constants.h | shape tags without `debugName()` | it returns `std::string`, which libcu++ does not have |
| NumTraits.h | no `NumTraits<std::string>` | same |
| util/Memory.h | `nvrtc_realloc` (malloc + memcpy + free, using the known old size) and `nvrtc_memmove` (overlap-safe byte copy), used where Eigen calls `realloc`/`memmove` | the device runtime has neither `realloc` nor `memmove` |
| RandomImpl.h | `eigen_random_device` with constant RAND_MAX-like values and a `__trap()`ing `run()` | no `std::rand`; `random_impl<bool>` is not a template, so the code must compile |
| products/GeneralBlockPanelKernel.h | `manage_caching_sizes` returns the default sizes | the original uses a function-scope static whose constructor queries the CPU |
| arch/Default/Half.h, BFloat16.h | `numeric_limits` specializations use `EIGEN_STD_NS_BEGIN/END` | specializations must be in cuda::std |
| util/DisableStupidWarnings.h | apply the nvcc diagnostic suppressions for NVRTC too, plus 1406 (trigraph message) | same front-end, same spurious warnings |

## Testing

`nvrtc-test/` is a standalone CMake project (C++17, `CUDA::nvrtc`,
`CUDA::cuda_driver`):

    cmake -S nvrtc-test -B build-nvrtc-test && cmake --build build-nvrtc-test
    ./build-nvrtc-test/eigen_nvrtc_test        # -v prints the NVRTC logs

`tests.cuh` holds the test functions. The program compiles it with NVRTC for
the architecture of device 0 using three option sets (C++17 and C++20 with
`-default-device`, C++17 without it, which is expected to fail), launches one
kernel per test, and compares the results with the same functions compiled
for the host (relative tolerance 1e-5, 1e-2 for half/bfloat16). It prints
PASS/FAIL per test and returns non-zero on failure.

Non-NVRTC regression checks done for this port: the host test subset
basicstuff, adjoint, array_cwise, product_small, lu, qr, cholesky,
geo_quaternion, half_float, bfloat16_float, numext, meta, indexed_view,
symbolic_index and nullary (129 ctest entries, g++ 11.4) and the nvcc tests
gpu_basic and gpu_example (CUDA 12.9, sm_120) all pass.
