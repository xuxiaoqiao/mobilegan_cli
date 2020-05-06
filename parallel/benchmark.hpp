#ifndef PARALLEL_BENCHMARK_HPP
#define PARALLEL_BENCHMARK_HPP

#include <CL/cl.h>
#include <vector>
#include "parallel/layers.hpp"
namespace parallel{

class Env {
 public:
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_device_id device = nullptr;
  static Env *Get() {
    static Env singleton;
    return &singleton;
  }
  Env();
};

class BenchmarkBase {
 protected:
  Env *env;
 public:
  BenchmarkBase() {
    env = Env::Get();
  }
};

class BenchmarkConv : public BenchmarkBase {
 private:
  static constexpr int CHANNEL = 256;
  static constexpr int OUT_HW = 64;
  static constexpr int IN_HW = OUT_HW + 2;
  std::vector<float> input;
  std::vector<float> weight;
  std::vector<float> bias;
  std::vector<float> output;
  cl_mem input_clbuf, weight_clbuf, bias_clbuf, output_clbuf;
 public:
  BenchmarkConv();
  void run(conv2d_variant var, int num_iteration);
 private:
  void doRun(conv2d_variant var);
};

void benchmark_run();
}

#endif //PARALLEL_BENCHMARK_HPP
