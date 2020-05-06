#include "parallel/benchmark.hpp"
#include "parallel/runtime.hpp"
#include "parallel/layers.hpp"
#include <CL/cl.h>
#include <iostream>
#include <vector>


namespace parallel {

Env::Env() {
  cl_int errNum;
  context = CreateContext();
  if (context == nullptr) {
    std::cerr << "Failed to create OpenCL context." << std::endl;
    std::abort();
  }
  queue = CreateCommandQueue(context, &device);
  if (queue == nullptr) {
    std::cerr << "Failed to create OpenCL command queue." << std::endl;
    std::abort();
  }
  init_kernels(context, device);
}


BenchmarkConv::BenchmarkConv() {
  input = std::vector<float>(IN_HW * IN_HW * CHANNEL, 1.0);
  weight = std::vector<float>(3 * 3 * CHANNEL * CHANNEL, 1.0);
  bias = std::vector<float>(CHANNEL, 1.0);
  output = std::vector<float>(OUT_HW * OUT_HW * CHANNEL, 1.0);
  input_clbuf = clCreateBuffer(env->context,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * IN_HW * IN_HW * CHANNEL,
                               input.data(),
                               NULL);
  weight_clbuf = clCreateBuffer(env->context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * 3 * 3 * CHANNEL * CHANNEL,
                                weight.data(),
                                NULL);
  bias_clbuf = clCreateBuffer(env->context,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * CHANNEL,
                              bias.data(),
                              NULL);
  output_clbuf = clCreateBuffer(env->context,
                                CL_MEM_READ_WRITE,
                                sizeof(float) * OUT_HW * OUT_HW * CHANNEL,
                                nullptr,
                                nullptr);
  if (!input_clbuf | !output_clbuf) {
    std::cout << "error clCreateBuffer" << std::endl;
    std::abort();
  }
}

void BenchmarkConv::doRun(conv2d_variant var) {
  conv2d_experimental_exec(env->queue,
                           input_clbuf,
                           weight_clbuf,
                           bias_clbuf,
                           output_clbuf,
                           nullptr,
                           nullptr,
                           CHANNEL,
                           IN_HW,
                           IN_HW,
                           CHANNEL,
                           OUT_HW,
                           OUT_HW,
                           1,
                           3,
                           3,
                           var);
}

void BenchmarkConv::run(conv2d_variant var, int num_iteration) {
  std::string label;
  switch (var) {
    case conv2d_variant::NCHW: label = "NCHW";
      break;
    case conv2d_variant::NCHW4: label = "NCHW4";
      break;
    case conv2d_variant::NHWC: label = "NHWC";
      break;
  }
  std::chrono::time_point<std::chrono::system_clock> phase_start;
  phase_start = std::chrono::system_clock::now();
  auto showtime = [&phase_start](const std::string &label) {
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - phase_start;
    std::cout << elapsed_seconds.count() << " s for " << label << std::endl;
    phase_start = std::chrono::system_clock::now();
  };
  for (int i = 0; i < num_iteration; i++) {
    doRun(var);
    showtime(label);
  }
}

void benchmark_run() {
  BenchmarkConv bench;
  bench.run(conv2d_variant::NHWC, 3);
  bench.run(conv2d_variant::NCHW4, 3);
  bench.run(conv2d_variant::NCHW, 3);

}
}
