#include "parallel/app.hpp"
#include "parallel/runtime.hpp"
#include "parallel/layers.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>

using namespace std;

namespace parallel {

int run(cl_mem input, cl_mem output, gan_buffer_t &buf, model &cycleGAN,
        cl_command_queue queue) {
  std::chrono::time_point<std::chrono::system_clock> start, phase_start;
  start = std::chrono::system_clock::now();
  phase_start = std::chrono::system_clock::now();
  auto showtime = [&phase_start](const char *label) {
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - phase_start;
    std::cout << elapsed_seconds.count() << " s for " << label << std::endl;
    phase_start = std::chrono::system_clock::now();
  };

  reflectpad_2d(queue, input, buf.o_0, 4/4, 256, 256, 3);
  conv2d_exec_async(queue, buf.o_0, cycleGAN.m_1.weight_buf,
                    cycleGAN.m_1.bias_buf, buf.o_1, cycleGAN.m_2.mean_buf,
                    cycleGAN.m_2.variance_buf, 4/4, 262, 262, 64/4,
                    256, 256,1, 7, 7, true,
                    activation::RELU);
  showtime("conv2d");

  zeropad_2d_onepix(queue, buf.o_1, buf.o_1_p, 64/4, 256, 256);
  conv2d_exec_async(queue, buf.o_1_p, cycleGAN.m_4.weight_buf,
                  cycleGAN.m_4.bias_buf, buf.o_2, cycleGAN.m_5.mean_buf,
                  cycleGAN.m_5.variance_buf, 64/4, 258, 258, 128/4,
                  128, 128,2, 3, 3, true,
                  activation::RELU);
  showtime("conv2d");

  zeropad_2d_onepix(queue, buf.o_2, buf.o_2_p, 128/4, 128, 128);
  conv2d_exec_async(queue, buf.o_2_p, cycleGAN.m_7.weight_buf,
                    cycleGAN.m_7.bias_buf, buf.o_3, cycleGAN.m_8.mean_buf,
                    cycleGAN.m_8.variance_buf, 128/4, 130, 130, 256/4,
                    64, 64,2, 3, 3, true,
                    activation::RELU);
  showtime("conv2d");

  // resnet block 0
  reflectpad_2d(queue, buf.o_3, buf.r0_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r0_0, cycleGAN.m_10.r_1.weight_buf,
                    cycleGAN.m_10.r_1.bias_buf, buf.r0_1, cycleGAN.m_10.r_2.mean_buf,
                    cycleGAN.m_10.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r0_1, buf.r0_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r0_2, cycleGAN.m_10.r_5.weight_buf,
                    cycleGAN.m_10.r_5.bias_buf, buf.r0_3, cycleGAN.m_10.r_6.mean_buf,
                    cycleGAN.m_10.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.o_3, buf.r0_3, 256*64*64);
  showtime("resnet0");

  // resnet block 1
  reflectpad_2d(queue, buf.r0_3, buf.r1_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r1_0, cycleGAN.m_11.r_1.weight_buf,
                    cycleGAN.m_11.r_1.bias_buf, buf.r1_1, cycleGAN.m_11.r_2.mean_buf,
                    cycleGAN.m_11.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r1_1, buf.r1_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r1_2, cycleGAN.m_11.r_5.weight_buf,
                    cycleGAN.m_11.r_5.bias_buf, buf.r1_3, cycleGAN.m_11.r_6.mean_buf,
                    cycleGAN.m_11.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r0_3, buf.r1_3, 256*64*64);
  showtime("resnet1");

  // resnet block 2
  reflectpad_2d(queue, buf.r1_3, buf.r2_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r2_0, cycleGAN.m_12.r_1.weight_buf,
                    cycleGAN.m_12.r_1.bias_buf, buf.r2_1, cycleGAN.m_12.r_2.mean_buf,
                    cycleGAN.m_12.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r2_1, buf.r2_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r2_2, cycleGAN.m_12.r_5.weight_buf,
                    cycleGAN.m_12.r_5.bias_buf, buf.r2_3, cycleGAN.m_12.r_6.mean_buf,
                    cycleGAN.m_12.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r1_3, buf.r2_3, 256*64*64);
  showtime("resnet2");

  // resnet block 3
  reflectpad_2d(queue, buf.r2_3, buf.r3_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r3_0, cycleGAN.m_13.r_1.weight_buf,
                    cycleGAN.m_13.r_1.bias_buf, buf.r3_1, cycleGAN.m_13.r_2.mean_buf,
                    cycleGAN.m_13.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r3_1, buf.r3_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r3_2, cycleGAN.m_13.r_5.weight_buf,
                    cycleGAN.m_13.r_5.bias_buf, buf.r3_3, cycleGAN.m_13.r_6.mean_buf,
                    cycleGAN.m_13.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r2_3, buf.r3_3, 256*64*64);
  showtime("resnet3");

  // resnet block 4
  reflectpad_2d(queue, buf.r3_3, buf.r4_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r4_0, cycleGAN.m_14.r_1.weight_buf,
                    cycleGAN.m_14.r_1.bias_buf, buf.r4_1, cycleGAN.m_14.r_2.mean_buf,
                    cycleGAN.m_14.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r4_1, buf.r4_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r4_2, cycleGAN.m_14.r_5.weight_buf,
                    cycleGAN.m_14.r_5.bias_buf, buf.r4_3, cycleGAN.m_14.r_6.mean_buf,
                    cycleGAN.m_14.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r3_3, buf.r4_3, 256*64*64);
  showtime("resnet4");

  // resnet block 5
  reflectpad_2d(queue, buf.r4_3, buf.r5_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r5_0, cycleGAN.m_15.r_1.weight_buf,
                    cycleGAN.m_15.r_1.bias_buf, buf.r5_1, cycleGAN.m_15.r_2.mean_buf,
                    cycleGAN.m_15.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r5_1, buf.r5_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r5_2, cycleGAN.m_15.r_5.weight_buf,
                    cycleGAN.m_15.r_5.bias_buf, buf.r5_3, cycleGAN.m_15.r_6.mean_buf,
                    cycleGAN.m_15.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r4_3, buf.r5_3, 256*64*64);
  showtime("resnet5");

  // resnet block 6
  reflectpad_2d(queue, buf.r5_3, buf.r6_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r6_0, cycleGAN.m_16.r_1.weight_buf,
                    cycleGAN.m_16.r_1.bias_buf, buf.r6_1, cycleGAN.m_16.r_2.mean_buf,
                    cycleGAN.m_16.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r6_1, buf.r6_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r6_2, cycleGAN.m_16.r_5.weight_buf,
                    cycleGAN.m_16.r_5.bias_buf, buf.r6_3, cycleGAN.m_16.r_6.mean_buf,
                    cycleGAN.m_16.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r5_3, buf.r6_3, 256*64*64);
  showtime("resnet6");

  // resnet block 7
  reflectpad_2d(queue, buf.r6_3, buf.r7_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r7_0, cycleGAN.m_17.r_1.weight_buf,
                    cycleGAN.m_17.r_1.bias_buf, buf.r7_1, cycleGAN.m_17.r_2.mean_buf,
                    cycleGAN.m_17.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r7_1, buf.r7_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r7_2, cycleGAN.m_17.r_5.weight_buf,
                    cycleGAN.m_17.r_5.bias_buf, buf.r7_3, cycleGAN.m_17.r_6.mean_buf,
                    cycleGAN.m_17.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r6_3, buf.r7_3, 256*64*64);
  showtime("resnet7");

  // resnet block 8
  reflectpad_2d(queue, buf.r7_3, buf.r8_0, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r8_0, cycleGAN.m_18.r_1.weight_buf,
                    cycleGAN.m_18.r_1.bias_buf, buf.r8_1, cycleGAN.m_18.r_2.mean_buf,
                    cycleGAN.m_18.r_2.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::RELU);
  reflectpad_2d(queue, buf.r8_1, buf.r8_2, 256/4, 64, 64, 1);
  conv2d_exec_async(queue, buf.r8_2, cycleGAN.m_18.r_5.weight_buf,
                    cycleGAN.m_18.r_5.bias_buf, buf.r8_3, cycleGAN.m_18.r_6.mean_buf,
                    cycleGAN.m_18.r_6.variance_buf, 256/4, 66, 66, 256/4,
                    64, 64,1, 3, 3, true,
                    activation::NONE);
  add(queue, buf.r7_3, buf.r8_3, 256*64*64);
  showtime("resnet8");

  // conv transpose
  conv2d_transpose_3x3_stride2_norm_relu_exec_async(queue, buf.r8_3, cycleGAN.m_19.weight_buf, cycleGAN.m_19.bias_buf,
      buf.o_4, cycleGAN.m_20.mean_buf, cycleGAN.m_20.variance_buf, 256/4, 64, 64,
      128/4, 128, 128, true, activation::RELU);
  showtime("conv transpose");

  conv2d_transpose_3x3_stride2_norm_relu_exec_async(queue, buf.o_4, cycleGAN.m_22.weight_buf, cycleGAN.m_22.bias_buf,
      buf.o_5, cycleGAN.m_23.mean_buf, cycleGAN.m_23.variance_buf, 128/4, 128, 128,
      64/4, 256, 256, true, activation::RELU);
  showtime("conv transpose");

  // conv
  reflectpad_2d(queue, buf.o_5, buf.o_6, 64/4, 256, 256, 3);
  conv2d_exec_async(queue, buf.o_6, cycleGAN.m_26.weight_buf, cycleGAN.m_26.bias_buf, output, nullptr,
                    nullptr, 64/4, 262, 262, 4/4,
                    256, 256,1, 7, 7, false,
                    activation::TANH);
  showtime("conv last");
  clFinish(queue);
  std::chrono::duration<double> total_elapsed_seconds = std::chrono::system_clock::now() - start;
  std::cout << total_elapsed_seconds.count() << " s for ALL" << std::endl;
  return 0;
}

gan_buffer_t::gan_buffer_t(cl_context context) {
  o_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 4 * 262 * 262, NULL, NULL);
  o_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 64 * 256 * 256, NULL, NULL);
  o_1_p = clCreateBuffer(context, CL_MEM_READ_WRITE,
                         sizeof(float) * 64 * 258 * 258, NULL, NULL);
  o_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 128 * 128 * 128, NULL, NULL);
  o_2_p = clCreateBuffer(context, CL_MEM_READ_WRITE,
                         sizeof(float) * 128 * 130 * 130, NULL, NULL);
  o_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r0_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r0_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r0_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r0_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r1_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r1_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r1_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r1_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r2_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r2_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r2_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r3_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r3_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r3_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r3_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r4_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r4_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r4_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r4_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r5_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r5_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r5_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r5_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r6_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r6_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r6_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r6_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r7_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r7_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r7_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r7_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r8_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r8_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  r8_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 66 * 66, NULL, NULL);
  r8_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 256 * 64 * 64, NULL, NULL);
  o_4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 128 * 128 * 128, NULL, NULL);
  o_5 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 64 * 256 * 256, NULL, NULL);
  o_6 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 64 * 262 * 262, NULL, NULL);
  bool allocation_failure =
      !o_0 || !o_1 || !o_1_p || !o_2 || !o_2_p || !o_3 || !r0_0 || !r0_1
          || !r0_2 || !r0_3 || !r1_0 || !r1_1 || !r1_2 || !r1_3 || !r2_0
          || !r2_1 || !r2_2 || !r2_3 || !r3_0 || !r3_1 || !r3_2 || !r3_3
          || !r4_0 || !r4_1 || !r4_2 || !r4_3 || !r5_0 || !r5_1 || !r5_2
          || !r5_3 || !r6_0 || !r6_1 || !r6_2 || !r6_3 || !r7_0 || !r7_1
          || !r7_2 || !r7_3 || !r8_0 || !r8_1 || !r8_2 || !r8_3 || !o_4 || !o_5
          || !o_6;
  if (allocation_failure) {
    std::cout << "allocation failure" << std::endl;
  }
}

gan_buffer_t::~gan_buffer_t() {
  clReleaseMemObject(o_0);
  clReleaseMemObject(o_1);
  clReleaseMemObject(o_1_p);
  clReleaseMemObject(o_2);
  clReleaseMemObject(o_2_p);
  clReleaseMemObject(o_3);
  clReleaseMemObject(r0_0);
  clReleaseMemObject(r0_1);
  clReleaseMemObject(r0_2);
  clReleaseMemObject(r0_3);
  clReleaseMemObject(r1_0);
  clReleaseMemObject(r1_1);
  clReleaseMemObject(r1_2);
  clReleaseMemObject(r1_3);
  clReleaseMemObject(r2_0);
  clReleaseMemObject(r2_1);
  clReleaseMemObject(r2_2);
  clReleaseMemObject(r2_3);
  clReleaseMemObject(r3_0);
  clReleaseMemObject(r3_1);
  clReleaseMemObject(r3_2);
  clReleaseMemObject(r3_3);
  clReleaseMemObject(r4_0);
  clReleaseMemObject(r4_1);
  clReleaseMemObject(r4_2);
  clReleaseMemObject(r4_3);
  clReleaseMemObject(r5_0);
  clReleaseMemObject(r5_1);
  clReleaseMemObject(r5_2);
  clReleaseMemObject(r5_3);
  clReleaseMemObject(r6_0);
  clReleaseMemObject(r6_1);
  clReleaseMemObject(r6_2);
  clReleaseMemObject(r6_3);
  clReleaseMemObject(r7_0);
  clReleaseMemObject(r7_1);
  clReleaseMemObject(r7_2);
  clReleaseMemObject(r7_3);
  clReleaseMemObject(r8_0);
  clReleaseMemObject(r8_1);
  clReleaseMemObject(r8_2);
  clReleaseMemObject(r8_3);
  clReleaseMemObject(o_4);
  clReleaseMemObject(o_5);
  clReleaseMemObject(o_6);
}

int example_main() {
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
  model gan_model;
  gan_buffer_t gan_buf(context);
  load_model(gan_model, context);
  std::cout << "Done load_model" << std::endl;

  constexpr int HEIGHT = 256;
  constexpr int WIDTH = 256;
  std::vector<float> input(HEIGHT * WIDTH * 4, 1.0);
  std::vector<float> output(HEIGHT * WIDTH * 4, 1.0);
  cl_mem input_clbuf = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * HEIGHT * WIDTH * 4,
                                      input.data(),
                                      NULL);
  cl_mem output_clbuf = clCreateBuffer(context,
                                       CL_MEM_READ_WRITE,
                                       sizeof(float) * HEIGHT * WIDTH * 4,
                                       nullptr,
                                       nullptr);
  if (!input_clbuf | !output_clbuf) {
    std::cout << "error clCreateBuffer" << std::endl;
    std::abort();
  }
  for (int i = 0; i < 10; i++) {
    run(input_clbuf, output_clbuf, gan_buf, gan_model, commandQueue);
    errNum = clEnqueueReadBuffer(commandQueue,
                                 output_clbuf,
                                 CL_TRUE,
                                 0,
                                 HEIGHT * WIDTH * 4 * sizeof(float),
                                 output.data(),
                                 0,
                                 NULL,
                                 NULL);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Error reading result buffer." << std::endl;
      std::abort();
    }
  }

  for (int i = 0; i < 100; i++) {
    std::cout << output.at(i) << ' ';
  }
  std::cout << std::endl;
  return 0;
}

}