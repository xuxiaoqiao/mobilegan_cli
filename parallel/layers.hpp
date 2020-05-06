#ifndef PARALLEL_LAYERS_HPP
#define PARALLEL_LAYERS_HPP

#include <CL/cl.h>

enum class activation {
  NONE,
  RELU,
  TANH
};

void init_kernels(cl_context context, cl_device_id device);


void convert_chw_to_chw4(cl_command_queue queue,
                         cl_mem input,
                         cl_mem output,
                         cl_long in_channel_num,
                         cl_long in_height,
                         cl_long in_width);

void convert_chw4_to_chw(cl_command_queue queue,
                         cl_mem input,
                         cl_mem output,
                         cl_long in_channel_num,
                         cl_long in_height,
                         cl_long in_width);

void conv2d_exec_async(
    cl_command_queue queue,
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
    activation act
    );

enum class conv2d_variant {
    NCHW4,
    NHWC,
    NHWC_VER2,
    NHWC_VER3,
    NCHW,
    NCHW_VER2,
    NCHW4_INTERLEAVE,
    NCHW4_INTERLEAVE_VER2,
    NCHW4_BLOCK
};
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
                              conv2d_variant variant);

void conv2d_transpose_3x3_stride2_norm_relu_exec_async(
    cl_command_queue queue,
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
    activation act
    );

void zeropad_2d_onepix(
    cl_command_queue queue,
    cl_mem input,
    cl_mem output,
    cl_long in_channel_num,
    cl_long in_height,
    cl_long in_width
    );

void reflectpad_2d(cl_command_queue queue,
                   cl_mem input,
                   cl_mem output,
                   cl_long in_channel_num,
                   cl_long in_height,
                   cl_long in_width,
                   cl_long padding);

void add(cl_command_queue queue, cl_mem src, cl_mem dst, cl_long len);

#endif // PARALLEL_LAYERS_HPP