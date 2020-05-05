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
                         cl_int in_channel_num,
                         cl_int in_height,
                         cl_int in_width);

void convert_chw4_to_chw(cl_command_queue queue,
                         cl_mem input,
                         cl_mem output,
                         cl_int in_channel_num,
                         cl_int in_height,
                         cl_int in_width);

void conv2d_exec_async(
    cl_command_queue queue,
    cl_mem input,
    cl_mem weight,
    cl_mem bias,
    cl_mem output,
    cl_mem mean,
    cl_mem variance,
    cl_int in_channel_num,
    cl_int in_height,
    cl_int in_width,
    cl_int out_channel_num,
    cl_int out_height,
    cl_int out_width,
    cl_int stride,
    cl_int kernel_height,
    cl_int kernel_width,
    bool fuse_instance_norm,
    activation act
    );


void conv2d_transpose_3x3_stride2_norm_relu_exec_async(
    cl_command_queue queue,
    cl_mem input,
    cl_mem weight,
    cl_mem bias,
    cl_mem output,
    cl_mem mean,
    cl_mem variance,
    cl_int in_channel_num,
    cl_int in_height,
    cl_int in_width,
    cl_int out_channel_num,
    cl_int out_height,
    cl_int out_width,
    bool fuse_instance_norm,
    activation act
    );

void zeropad_2d_onepix(
    cl_command_queue queue,
    cl_mem input,
    cl_mem output,
    cl_int in_channel_num,
    cl_int in_height,
    cl_int in_width
    );

void reflectpad_2d(cl_command_queue queue,
                   cl_mem input,
                   cl_mem output,
                   cl_int in_channel_num,
                   cl_int in_height,
                   cl_int in_width,
                   cl_int padding);

void add(cl_command_queue queue, cl_mem src, cl_mem dst, cl_int len);

#endif // PARALLEL_LAYERS_HPP