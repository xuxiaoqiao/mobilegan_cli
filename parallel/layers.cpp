#include <iostream>
#include <array>
#include "parallel/layers.hpp"
#include "parallel/runtime.hpp"


static bool initialized = false;
static cl_program conv2d_regular_program = nullptr;
static cl_kernel conv2d_regular_kernel = nullptr;
static cl_program conv2d_norm_program = nullptr;
static cl_kernel conv2d_norm_kernel = nullptr;
static cl_program conv2d_norm_relu_program = nullptr;
static cl_kernel conv2d_norm_relu_kernel = nullptr;
static cl_program conv2d_tanh_program = nullptr;
static cl_kernel conv2d_tanh_kernel = nullptr;

static cl_program conv2d_exp_program = nullptr;
static cl_kernel conv2d_nchw_kernel = nullptr;
static cl_kernel conv2d_nchw_ver2_kernel = nullptr;
static cl_kernel conv2d_nhwc_kernel = nullptr;
static cl_kernel conv2d_nchw4_interleave_kernel = nullptr;
static cl_kernel conv2d_nchw4_block_kernel = nullptr;

static cl_program conv2d_transpose_regular_program = nullptr;
static cl_kernel conv2d_transpose_regular_kernel = nullptr;
static cl_program conv2d_transpose_norm_relu_program = nullptr;
static cl_kernel conv2d_transpose_norm_relu_kernel = nullptr;

static cl_program conversion_program = nullptr;
static cl_kernel convert_chw_to_chw4_kernel = nullptr;
static cl_kernel convert_chw4_to_chw_kernel = nullptr;

static cl_program pad_program = nullptr;
static cl_kernel zero_pad_2d_onepix_kernel = nullptr;
static cl_kernel reflect_pad_2d_onepix_kernel = nullptr;
static cl_kernel reflect_pad_2d_general_kernel = nullptr;

static cl_program util_program = nullptr;
static cl_kernel add_kernel = nullptr;

void init_kernels(cl_context context, cl_device_id device) {
  // TODO: initialize all the cl_program & cl_kernel
  if (initialized) {
    std::cout << "already initialized" << std::endl;
    return;
  }
  initialized = true;
  conv2d_regular_program = CreateProgram(context,
                                         device,
                                         "conv2d.cl",
                                         "");
  conv2d_regular_kernel = clCreateKernel(conv2d_regular_program, "conv2d", nullptr);

  conv2d_norm_program = CreateProgram(context,
                                           device,
                                           "conv2d.cl",
                                           "-DUSE_INSTANCE_NORM=1");
  conv2d_norm_kernel =
      clCreateKernel(conv2d_norm_program, "conv2d", nullptr);

  conv2d_norm_relu_program = CreateProgram(context,
                                           device,
                                           "conv2d.cl",
                                           "-DUSE_INSTANCE_NORM=1 -DUSE_RELU=1");
  conv2d_norm_relu_kernel =
      clCreateKernel(conv2d_norm_relu_program, "conv2d", nullptr);

  conv2d_tanh_program = CreateProgram(context,
                                      device,
                                      "conv2d.cl",
                                      "-DUSE_TANH=1");
  conv2d_tanh_kernel = clCreateKernel(conv2d_tanh_program, "conv2d", nullptr);

  conv2d_transpose_regular_program = CreateProgram(context,
                                                   device,
                                                   "conv2d_transpose.cl",
                                                   "");
  conv2d_transpose_regular_kernel = clCreateKernel(
      conv2d_transpose_regular_program,
      "conv2d_transpose_3x3_stride2",
      nullptr);
  conv2d_transpose_norm_relu_program = CreateProgram(context,
                                                   device,
                                                   "conv2d_transpose.cl",
                                                   "-DUSE_INSTANCE_NORM=1 -DUSE_RELU=1");
  conv2d_transpose_norm_relu_kernel = clCreateKernel(
      conv2d_transpose_norm_relu_program,
      "conv2d_transpose_3x3_stride2",
      nullptr);

  conv2d_exp_program = CreateProgram(context,
                                         device,
                                         "conv2d_exp.cl",
                                         "");
  conv2d_nchw_kernel = clCreateKernel(conv2d_exp_program, "conv2d_nchw", nullptr);
  conv2d_nhwc_kernel = clCreateKernel(conv2d_exp_program, "conv2d_nhwc", nullptr);
  conv2d_nchw4_interleave_kernel = clCreateKernel(conv2d_exp_program, "conv2d_nchw4_interleave", nullptr);
  conv2d_nchw4_block_kernel = clCreateKernel(conv2d_exp_program, "conv2d_nchw4_block", nullptr);
  conv2d_nchw_ver2_kernel =  clCreateKernel(conv2d_exp_program, "conv2d_nchw_ver2", nullptr);

  conversion_program = CreateProgram(context,
                                     device,
                                     "conversion.cl",
                                     "");
  convert_chw_to_chw4_kernel = clCreateKernel(
      conversion_program,
      "conv_chw_to_chw4",
      nullptr);
  convert_chw4_to_chw_kernel = clCreateKernel(
      conversion_program,
      "conv_chw4_to_chw",
      nullptr);

  pad_program = CreateProgram(context, device, "pad.cl", "");
  zero_pad_2d_onepix_kernel = clCreateKernel(
      pad_program,
      "zero_pad_2d_onepix",
      nullptr);
  reflect_pad_2d_onepix_kernel = clCreateKernel(
      pad_program,
      "reflect_pad_2d_onepix",
      nullptr);
  reflect_pad_2d_general_kernel = clCreateKernel(
      pad_program,
      "reflect_pad_2d",
      nullptr);

  util_program = CreateProgram(context, device, "util.cl", "");
  add_kernel = clCreateKernel(util_program, "add", nullptr);
}

void conv2d_exec_async(cl_command_queue queue,
                       cl_mem input,
                       cl_mem weight,
                       cl_mem bias,
                       cl_mem output,
                       cl_mem mean,
                       cl_mem variance,
                       cl_long in_channel_num,
                       cl_long in_height,
                       cl_long in_width,
                       cl_long out_channel_num,
                       cl_long out_height,
                       cl_long out_width,
                       cl_long stride,
                       cl_long kernel_height,
                       cl_long kernel_width,
                       bool fuse_instance_norm,
                       activation act) {
  cl_kernel kernel = nullptr;
  if (fuse_instance_norm) {
    assert(act == activation::RELU || act == activation::NONE);
    if (act == activation::RELU) {
      kernel = conv2d_norm_relu_kernel;
    } else {
      kernel = conv2d_norm_kernel;
    }
  } else {
    if (act == activation::TANH) {
      kernel = conv2d_tanh_kernel;
    } else {
      kernel = conv2d_regular_kernel;
    }
  }
  cl_int errNum;
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias);
  errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
  errNum |= clSetKernelArg(kernel, 4, sizeof(cl_long), &in_channel_num);
  errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long), &in_height);
  errNum |= clSetKernelArg(kernel, 6, sizeof(cl_long), &in_width);
  errNum |= clSetKernelArg(kernel, 7, sizeof(cl_long), &out_channel_num);
  errNum |= clSetKernelArg(kernel, 8, sizeof(cl_long), &out_height);
  errNum |= clSetKernelArg(kernel, 9, sizeof(cl_long), &out_width);
  errNum |= clSetKernelArg(kernel, 10, sizeof(cl_long), &stride);
  errNum |= clSetKernelArg(kernel, 11, sizeof(cl_long), &kernel_height);
  errNum |= clSetKernelArg(kernel, 12, sizeof(cl_long), &kernel_width);
  if (fuse_instance_norm) {
    errNum |= clSetKernelArg(kernel, 13, sizeof(cl_mem), &mean);
    errNum |= clSetKernelArg(kernel, 14, sizeof(cl_mem), &variance);
  }
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    std::abort();
  }

  size_t output_channel_blk = (out_channel_num + 3) / 4;
  size_t globalWorkSize[3] = {output_channel_blk, (size_t) out_height,
                              (size_t) out_width / 2}; // (C_block, H, W/UNROLL)

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr,
                                  globalWorkSize, nullptr,
                                  0, nullptr, nullptr);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::abort();
  }
  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}


void conv2d_experimental_exec(cl_command_queue queue,
                       cl_mem input,
                       cl_mem weight,
                       cl_mem bias,
                       cl_mem output,
                       cl_mem mean,
                       cl_mem variance,
                       cl_long in_channel_num,
                       cl_long in_height,
                       cl_long in_width,
                       cl_long out_channel_num,
                       cl_long out_height,
                       cl_long out_width,
                       cl_long stride,
                       cl_long kernel_height,
                       cl_long kernel_width,
                       conv2d_variant variant) {
  cl_kernel kernel = nullptr;
  switch (variant) {
    case conv2d_variant::NHWC: kernel = conv2d_nhwc_kernel;
      break;
    case conv2d_variant::NCHW: kernel = conv2d_nchw_kernel;
      break;
    case conv2d_variant::NCHW4: kernel = conv2d_regular_kernel;
      break;
    case conv2d_variant::NCHW4_INTERLEAVE: kernel = conv2d_nchw4_interleave_kernel;
      break;
    case conv2d_variant::NCHW4_BLOCK: kernel = conv2d_nchw4_block_kernel;
      break;
    case conv2d_variant::NCHW_VER2: kernel = conv2d_nchw_ver2_kernel;
      break;
  }
  cl_int errNum;
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias);
  errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
  errNum |= clSetKernelArg(kernel, 4, sizeof(cl_long), &in_channel_num);
  errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long), &in_height);
  errNum |= clSetKernelArg(kernel, 6, sizeof(cl_long), &in_width);
  errNum |= clSetKernelArg(kernel, 7, sizeof(cl_long), &out_channel_num);
  errNum |= clSetKernelArg(kernel, 8, sizeof(cl_long), &out_height);
  errNum |= clSetKernelArg(kernel, 9, sizeof(cl_long), &out_width);
  errNum |= clSetKernelArg(kernel, 10, sizeof(cl_long), &stride);
  errNum |= clSetKernelArg(kernel, 11, sizeof(cl_long), &kernel_height);
  errNum |= clSetKernelArg(kernel, 12, sizeof(cl_long), &kernel_width);

  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    std::abort();
  }

  std::array<size_t, 3> globalWorkSize;
  switch (variant) {
    case conv2d_variant::NCHW4:
    case conv2d_variant::NCHW4_BLOCK:
    case conv2d_variant::NCHW4_INTERLEAVE:
      globalWorkSize = {(size_t) (out_channel_num + 3)/4, (size_t) out_height,
                        (size_t) out_width/2}; // (C_block, H, W/UNROLL)
      break;
    case conv2d_variant::NHWC:
      globalWorkSize = {(size_t) out_height,
                        (size_t) out_width,
                        (size_t) out_channel_num};
      break;
    case conv2d_variant::NCHW:
      globalWorkSize = {(size_t) out_channel_num, (size_t) out_height,
                        (size_t) out_width};
      break;
    case conv2d_variant::NCHW_VER2:
      globalWorkSize = {(size_t) out_channel_num, (size_t) out_height,
                        (size_t) out_width / 4};
      break;
  }


  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr,
                                  globalWorkSize.data(), nullptr,
                                  0, nullptr, nullptr);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::abort();
  }
  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}


void conv2d_transpose_3x3_stride2_norm_relu_exec_async(cl_command_queue queue,
                                                       cl_mem input,
                                                       cl_mem weight,
                                                       cl_mem bias,
                                                       cl_mem output,
                                                       cl_mem mean,
                                                       cl_mem variance,
                                                       cl_long in_channel_num,
                                                       cl_long in_height,
                                                       cl_long in_width,
                                                       cl_long out_channel_num,
                                                       cl_long out_height,
                                                       cl_long out_width,
                                                       bool fuse_instance_norm,
                                                       activation act) {
  cl_kernel kernel;
  if (fuse_instance_norm) {
    assert(act == activation::RELU);
    kernel = conv2d_transpose_norm_relu_kernel;
  } else {
    assert(act == activation::NONE);
    kernel = conv2d_transpose_regular_kernel;
  }
  cl_int errNum;
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias);
  errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
  errNum |= clSetKernelArg(kernel, 4, sizeof(cl_long), &in_channel_num);
  errNum |= clSetKernelArg(kernel, 5, sizeof(cl_long), &in_height);
  errNum |= clSetKernelArg(kernel, 6, sizeof(cl_long), &in_width);
  errNum |= clSetKernelArg(kernel, 7, sizeof(cl_long), &out_channel_num);
  errNum |= clSetKernelArg(kernel, 8, sizeof(cl_long), &out_height);
  errNum |= clSetKernelArg(kernel, 9, sizeof(cl_long), &out_width);
  errNum |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &mean);
  errNum |= clSetKernelArg(kernel, 11, sizeof(cl_mem), &variance);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    std::abort();
  }

  size_t output_channel_blk = (out_channel_num + 3) / 4;
  size_t globalWorkSize[3] = {output_channel_blk, (size_t) out_height,
                              (size_t) out_width * 4}; // (C_block, H, W/UNROLL)

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr,
                                  globalWorkSize, nullptr,
                                  0, nullptr, nullptr);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::abort();
  }
  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}

void convert_chw_to_chw4(cl_command_queue queue,
                         cl_mem input,
                         cl_mem output,
                         cl_long in_channel_num,
                         cl_long in_height,
                         cl_long in_width) {
  assert(in_channel_num == 3 || in_channel_num % 4 == 0);
  cl_int errNum;
  errNum = clSetKernelArg(convert_chw_to_chw4_kernel, 0, sizeof(cl_mem), &input);
  errNum |= clSetKernelArg(convert_chw_to_chw4_kernel, 1, sizeof(cl_mem), &output);
  errNum |= clSetKernelArg(convert_chw_to_chw4_kernel, 2, sizeof(cl_int), &in_channel_num);
  errNum |= clSetKernelArg(convert_chw_to_chw4_kernel, 3, sizeof(cl_int), &in_height);
  errNum |= clSetKernelArg(convert_chw_to_chw4_kernel, 4, sizeof(cl_int), &in_width);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    std::abort();
  }

  size_t output_channel_blk = (in_channel_num + 3) / 4;
  size_t globalWorkSize[3] = {output_channel_blk, (size_t) in_height,
                              (size_t) in_width};

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, convert_chw_to_chw4_kernel, 3, nullptr,
                                  globalWorkSize, nullptr,
                                  0, nullptr, nullptr);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::abort();
  }
  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}

void convert_chw4_to_chw(cl_command_queue queue,
                         cl_mem input,
                         cl_mem output,
                         cl_long in_channel_num,
                         cl_long in_height,
                         cl_long in_width) {
  assert(in_channel_num == 3 || in_channel_num % 4 == 0);
  cl_int errNum;
  errNum = clSetKernelArg(convert_chw4_to_chw_kernel, 0, sizeof(cl_mem), &input);
  errNum |= clSetKernelArg(convert_chw4_to_chw_kernel, 1, sizeof(cl_mem), &output);
  errNum |= clSetKernelArg(convert_chw4_to_chw_kernel, 2, sizeof(cl_long), &in_channel_num);
  errNum |= clSetKernelArg(convert_chw4_to_chw_kernel, 3, sizeof(cl_long), &in_height);
  errNum |= clSetKernelArg(convert_chw4_to_chw_kernel, 4, sizeof(cl_long), &in_width);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    std::abort();
  }

  size_t globalWorkSize[3] = {(size_t) in_channel_num, (size_t) in_height,
                              (size_t) in_width};

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, convert_chw4_to_chw_kernel, 3, nullptr,
                                  globalWorkSize, nullptr,
                                  0, nullptr, nullptr);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::abort();
  }
  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}

void zeropad_2d_onepix(cl_command_queue queue,
                       cl_mem input,
                       cl_mem output,
                       cl_long in_channel_num,
                       cl_long in_height,
                       cl_long in_width) {
  cl_int errNum;
  errNum = clSetKernelArg(zero_pad_2d_onepix_kernel, 0, sizeof(cl_mem), &input);
  errNum |= clSetKernelArg(zero_pad_2d_onepix_kernel, 1, sizeof(cl_mem), &output);
  errNum |= clSetKernelArg(zero_pad_2d_onepix_kernel, 2, sizeof(cl_long), &in_channel_num);
  errNum |= clSetKernelArg(zero_pad_2d_onepix_kernel, 3, sizeof(cl_long), &in_height);
  errNum |= clSetKernelArg(zero_pad_2d_onepix_kernel, 4, sizeof(cl_long), &in_width);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    std::abort();
  }

  size_t globalWorkSize[3] = {(size_t) (in_channel_num + 3) / 4, (size_t) in_height,
                              (size_t) in_width / 2};

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, zero_pad_2d_onepix_kernel, 3, nullptr,
                                  globalWorkSize, nullptr,
                                  0, nullptr, nullptr);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::abort();
  }
  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}

void reflectpad_2d(cl_command_queue queue,
                   cl_mem input,
                   cl_mem output,
                   cl_long in_channel_num,
                   cl_long in_height,
                   cl_long in_width,
                   cl_long padding) {
  cl_int errNum;
  if (padding == 1) {
    errNum = clSetKernelArg(reflect_pad_2d_onepix_kernel, 0, sizeof(cl_mem), &input);
    errNum |= clSetKernelArg(reflect_pad_2d_onepix_kernel, 1, sizeof(cl_mem), &output);
    errNum |= clSetKernelArg(reflect_pad_2d_onepix_kernel, 2, sizeof(cl_long), &in_channel_num);
    errNum |= clSetKernelArg(reflect_pad_2d_onepix_kernel, 3, sizeof(cl_long), &in_height);
    errNum |= clSetKernelArg(reflect_pad_2d_onepix_kernel, 4, sizeof(cl_long), &in_width);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Error setting kernel arguments." << std::endl;
      std::abort();
    }

    size_t globalWorkSize[3] = {(size_t) (in_channel_num + 3) / 4, (size_t) in_height,
                                (size_t) in_width / 2};

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(queue, reflect_pad_2d_onepix_kernel, 3, nullptr,
                                    globalWorkSize, nullptr,
                                    0, nullptr, nullptr);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Error queuing kernel for execution." << std::endl;
      std::abort();
    }
  } else {
    errNum = clSetKernelArg(reflect_pad_2d_general_kernel, 0, sizeof(cl_mem), &input);
    errNum |= clSetKernelArg(reflect_pad_2d_general_kernel, 1, sizeof(cl_mem), &output);
    errNum |= clSetKernelArg(reflect_pad_2d_general_kernel, 2, sizeof(cl_long), &in_channel_num);
    errNum |= clSetKernelArg(reflect_pad_2d_general_kernel, 3, sizeof(cl_long), &in_height);
    errNum |= clSetKernelArg(reflect_pad_2d_general_kernel, 4, sizeof(cl_long), &in_width);
    errNum |= clSetKernelArg(reflect_pad_2d_general_kernel, 5, sizeof(cl_long), &padding);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Error setting kernel arguments." << std::endl;
      std::abort();
    }

    size_t globalWorkSize[3] = {(size_t) (in_channel_num + 3) / 4, (size_t) in_height,
                                (size_t) in_width};

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(queue, reflect_pad_2d_general_kernel, 3, nullptr,
                                    globalWorkSize, nullptr,
                                    0, nullptr, nullptr);
    if (errNum != CL_SUCCESS) {
      std::cerr << "Error queuing kernel for execution." << std::endl;
      std::abort();
    }
  }

  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}

void add(cl_command_queue queue, cl_mem src, cl_mem dst, cl_long len) {
  cl_int errNum;
  errNum = clSetKernelArg(add_kernel, 0, sizeof(cl_mem), &src);
  errNum |= clSetKernelArg(add_kernel, 1, sizeof(cl_mem), &dst);
  errNum |= clSetKernelArg(add_kernel, 2, sizeof(cl_long), &len);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error setting kernel arguments." << std::endl;
    std::abort();
  }

  size_t globalWorkSize[1] = {(size_t) len};

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(queue, add_kernel, 1, nullptr,
                                  globalWorkSize, nullptr,
                                  0, nullptr, nullptr);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::abort();
  }
  errNum = clFinish(queue);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error clFinish()." << std::endl;
    std::abort();
  }
}
