// Eigen NVRTC validation harness.
//
// Compiles tests.cuh with NVRTC using several option sets, runs every test
// kernel on the GPU and compares the results with the same test functions
// compiled for the host.
#include <cuda.h>
#include <nvrtc.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tests.cuh"

#ifndef EIGEN_NVRTC_ROOT
#error "EIGEN_NVRTC_ROOT must be defined"
#endif
#ifndef CUDA_INCLUDE_DIR
#error "CUDA_INCLUDE_DIR must be defined"
#endif
#ifndef TESTS_SOURCE
#error "TESTS_SOURCE must be defined"
#endif

namespace {

constexpr int kIn = 64;
constexpr int kOut = 1024;

#define CU_CHECK(x)                                                         \
  do {                                                                      \
    CUresult r_ = (x);                                                      \
    if (r_ != CUDA_SUCCESS) {                                               \
      const char* s_ = nullptr;                                             \
      cuGetErrorString(r_, &s_);                                            \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, \
                   s_ ? s_ : "?");                                          \
      std::exit(2);                                                         \
    }                                                                       \
  } while (0)

struct Test {
  const char* name;
  void (*host)(const double*, double*);
  double tol;  // relative tolerance
};

#define EIGEN_NVRTC_TEST_ENTRY(name) {#name, &test_##name, 0.0},
std::vector<Test> make_tests() {
  std::vector<Test> t = {EIGEN_NVRTC_TESTS(EIGEN_NVRTC_TEST_ENTRY)};
  for (auto& x : t) {
    std::string n = x.name;
    x.tol = (n == "half") ? 1e-2 : 1e-5;  // float results are part of most tests
  }
  return t;
}

std::string read_file(const char* path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path);
    std::exit(2);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool compile(const std::string& src, const std::vector<std::string>& opts, std::string& ptx,
             std::string& log) {
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog, src.c_str(), "tests.cuh", 0, nullptr, nullptr);
  std::vector<const char*> o;
  for (auto& s : opts) o.push_back(s.c_str());
  nvrtcResult r = nvrtcCompileProgram(prog, (int)o.size(), o.data());
  size_t n = 0;
  nvrtcGetProgramLogSize(prog, &n);
  log.assign(n, '\0');
  nvrtcGetProgramLog(prog, &log[0]);
  if (r == NVRTC_SUCCESS) {
    nvrtcGetPTXSize(prog, &n);
    ptx.assign(n, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
  }
  nvrtcDestroyProgram(&prog);
  return r == NVRTC_SUCCESS;
}

bool close_enough(double a, double b, double tol) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  if (std::isinf(a) || std::isinf(b)) return a == b;
  return std::fabs(a - b) <= tol * std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = EIGEN_NVRTC_ROOT;
  const std::string cuda_inc = CUDA_INCLUDE_DIR;
  const std::string src = read_file(TESTS_SOURCE);
  const bool verbose = argc > 1 && std::strcmp(argv[1], "-v") == 0;

  std::vector<double> in(kIn);
  for (int i = 0; i < kIn; ++i) in[i] = std::sin(1.0 + 0.7 * i) * (1.0 + 0.1 * (i % 7));

  CU_CHECK(cuInit(0));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, 0));
  int major = 0, minor = 0;
  CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  CUcontext ctx;
#if CUDA_VERSION >= 13000
  CU_CHECK(cuCtxCreate(&ctx, nullptr, 0, dev));
#else
  CU_CHECK(cuCtxCreate(&ctx, 0, dev));
#endif
  const std::string arch = "--gpu-architecture=compute_" + std::to_string(major * 10 + minor);

  struct Config {
    const char* name;
    std::vector<std::string> extra;
  };
  const std::vector<Config> configs = {
      {"c++17 -default-device", {"-std=c++17", "-default-device"}},
      {"c++20 -default-device", {"-std=c++20", "-default-device"}},
      {"c++17 (no -default-device)", {"-std=c++17"}},
  };

  const std::vector<Test> tests = make_tests();
  int failures = 0;
  for (const Config& cfg : configs) {
    std::vector<std::string> opts = cfg.extra;
    opts.push_back(arch);
    opts.push_back("-I" + root);
    opts.push_back("-I" + cuda_inc);
    std::printf("=== NVRTC options: %s %s -I%s -I%s\n", cfg.name, arch.c_str(), root.c_str(),
                cuda_inc.c_str());
    std::string ptx, log;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = compile(src, opts, ptx, log);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
      // Count the errors and show the first lines of the log.
      int nerr = 0;
      std::istringstream ls(log);
      std::string line, first;
      while (std::getline(ls, line))
        if (line.find("error") != std::string::npos) {
          if (nerr++ < 3) first += "      " + line.substr(0, 240) + "\n";
        }
      std::printf("  COMPILE FAIL (%.1fs, %d error lines)\n%s", secs, nerr, first.c_str());
      if (verbose) std::printf("%s\n", log.c_str());
      if (std::string(cfg.name).find("no -default-device") == std::string::npos) ++failures;
      else std::printf("  (expected: Eigen requires -default-device under NVRTC)\n");
      continue;
    }
    std::printf("  compiled in %.1fs, PTX %zu bytes\n", secs, ptx.size());
    if (verbose && !log.empty()) std::printf("%s\n", log.c_str());

    CUmodule mod;
    CU_CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUdeviceptr d_in, d_out;
    CU_CHECK(cuMemAlloc(&d_in, kIn * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_out, kOut * sizeof(double)));
    CU_CHECK(cuMemcpyHtoD(d_in, in.data(), kIn * sizeof(double)));
    for (const Test& t : tests) {
      std::vector<double> ref(kOut, 0.0), got(kOut, 0.0);
      t.host(in.data(), ref.data());
      CUfunction f;
      CU_CHECK(cuModuleGetFunction(&f, mod, (std::string("kernel_") + t.name).c_str()));
      CU_CHECK(cuMemsetD8(d_out, 0, kOut * sizeof(double)));
      void* args[] = {&d_in, &d_out};
      CU_CHECK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
      CU_CHECK(cuCtxSynchronize());
      CU_CHECK(cuMemcpyDtoH(got.data(), d_out, kOut * sizeof(double)));
      const int n = (int)ref[0];
      int bad = 0, first_bad = -1;
      if ((int)got[0] != n) {
        bad = -1;
      } else {
        for (int i = 1; i <= n; ++i)
          if (!close_enough(ref[i], got[i], t.tol)) {
            if (first_bad < 0) first_bad = i - 1;
            ++bad;
          }
      }
      if (bad == 0) {
        std::printf("  PASS  %-8s (%d values)\n", t.name, n);
      } else {
        ++failures;
        if (bad < 0)
          std::printf("  FAIL  %-8s (result count %d, expected %d)\n", t.name, (int)got[0], n);
        else
          std::printf("  FAIL  %-8s (%d/%d mismatches, first #%d: host %.17g device %.17g)\n", t.name,
                      bad, n, first_bad, ref[first_bad + 1], got[first_bad + 1]);
      }
    }
    CU_CHECK(cuMemFree(d_in));
    CU_CHECK(cuMemFree(d_out));
    CU_CHECK(cuModuleUnload(mod));
  }
  CU_CHECK(cuCtxDestroy(ctx));
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
