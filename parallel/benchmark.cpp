#include "parallel/benchmark.hpp"
#include "parallel/runtime.hpp"
#include "parallel/layers.hpp"
#include <CL/cl.h>
#include <iostream>

namespace parallel {
void benchmark_run() {
  cl_context context = nullptr;
  cl_command_queue commandQueue = nullptr;
  cl_device_id device = nullptr;
  cl_int errNum;
  context = CreateContext();
  if (context == nullptr) {
    std::cerr << "Failed to create OpenCL context." << std::endl;
    std::abort();
  }
  commandQueue = CreateCommandQueue(context, &device);
  if (commandQueue == nullptr) {
    std::cerr << "Failed to create OpenCL command queue." << std::endl;
    std::abort();
  }

  init_kernels(context, device);
}
}
