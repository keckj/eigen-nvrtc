<p align="center">
  <img src="doc/logo.png" alt="eigen-nvrtc logo" width="260">
</p>

<h1 align="center">eigen-nvrtc</h1>

<p align="center">
  <b>Eigen 5.0.1, usable in CUDA kernels compiled at runtime with NVRTC.</b>
</p>

---

**Upstream Eigen does not compile under NVRTC**, the CUDA runtime compilation
library: it includes system headers that do not exist there (`<cmath>`,
`<type_traits>`, `<cassert>`, `<cuda.h>`, ...), calls host-only functions
(`realloc`, `memmove`, `std::rand`) and uses `std::string`. This fork fixes
that, so that the same `#include <Eigen/Dense>` works in code given to
`nvrtcCompileProgram`:

```cpp
// kernel.cu, compiled at runtime with NVRTC
#include <Eigen/Dense>

extern "C" __global__ void solve(const double* in, double* out)
{
    Eigen::Map<const Eigen::Matrix3d> A(in);
    Eigen::Vector3d b(in[9], in[10], in[11]);
    Eigen::Map<Eigen::Vector3d> x(out);
    x = A.partialPivLu().solve(b);
}
```

```cpp
// host code
const char* opts[] = {"-std=c++17", "-default-device", "--gpu-architecture=compute_120",
                      "-I/path/to/eigen-nvrtc", "-I/usr/local/cuda/include"};
nvrtcCreateProgram(&prog, source, "kernel.cu", 0, nullptr, nullptr);
nvrtcCompileProgram(prog, 5, opts);
```

No `-D` macro, no pre-included header: `-default-device` and the two include
paths are all that is needed.

## Highlights

- **Drop-in**: every change is guarded by `EIGEN_CUDACC_RTC`, defined only
  under NVRTC. With g++, clang, MSVC or nvcc, the preprocessed token stream is
  identical to pristine Eigen 5.0.1.
- **Small**: about 150 changed lines in 15 Eigen files, plus one new header
  ([Eigen/src/Core/util/NVRTCSupport.h](Eigen/src/Core/util/NVRTCSupport.h))
  that maps `std::` to libcu++ (`cuda::std`).
- **Broad coverage** under NVRTC: fixed and dynamic-size matrices, `Map` over
  device pointers, reductions, block/indexed views, LU, LLT, LDLT, QR,
  JacobiSVD, SelfAdjointEigenSolver, Geometry (quaternions, rotations,
  isometries), `half` and `bfloat16`, custom scalar types (such as automatic
  differentiation numbers).
- **Tested**: [nvrtc-test/](nvrtc-test) compiles all of the above with NVRTC
  (C++17 and C++20), runs it on the GPU and compares with the same code
  compiled for the host. Upstream host and nvcc tests were rerun and pass
  (see [NVRTC.md](NVRTC.md#testing)).

## Requirements

- CUDA 12.x NVRTC with the libcu++ (CCCL) headers (tested with CUDA 12.9 on
  sm_120);
- C++17 or later;
- the NVRTC option `-default-device` (Eigen leaves many functions without
  `__host__ __device__` annotations).

## Getting it

- Clone the `5.0.1-nvrtc` tag (or the `nvrtc-5.0.1` branch) of this
  repository and point NVRTC to its root:

  ```
  git clone --depth 1 --branch 5.0.1-nvrtc https://github.com/keckj/eigen-nvrtc.git
  ```

- Or, with CMake:

  ```cmake
  include(FetchContent)
  FetchContent_Declare(eigen_nvrtc
      GIT_REPOSITORY https://github.com/keckj/eigen-nvrtc.git
      GIT_TAG 5.0.1-nvrtc
      GIT_SHALLOW TRUE
      SOURCE_SUBDIR do-not-add-subdirectory) # headers only
  FetchContent_MakeAvailable(eigen_nvrtc)    # headers in ${eigen_nvrtc_SOURCE_DIR}
  ```

- Or apply `eigen-5.0.1-nvrtc.patch`, attached to the
  [release](https://github.com/keckj/eigen-nvrtc/releases/tag/5.0.1-nvrtc),
  to an Eigen 5.0.1 checkout: `git apply eigen-5.0.1-nvrtc.patch`.

## Documentation

[NVRTC.md](NVRTC.md) details the supported features, the limitations (no
random numbers, no I/O, device heap for dynamic sizes, ...), the design of the
`std` shim, the list of patched files and how to run the tests.

## Upstream Eigen

**Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.**

For more information go to http://eigen.tuxfamily.org/ or https://libeigen.gitlab.io/docs/.

For ***pull request***, ***bug reports***, and ***feature requests*** about Eigen itself, go to https://gitlab.com/libeigen/eigen.
Issues specific to NVRTC belong to this repository.

Eigen and this fork are licensed under the MPL2 (see [COPYING.README](COPYING.README)).
