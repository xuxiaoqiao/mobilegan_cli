#include "common.h"

// precondition: in_channel_num % 16 == 0 && out_channel_num % 16 == 0
// __kernel void conv2d_nhwc(
//   __global float* input, /* layout: HWC */
//   __global float* weight, /* layout: HWOI */
//   __global float* bias,
//   __global float* output, /* layout: HWC */
//   __private long in_channel_num,
//   __private long in_height,
//   __private long in_width,
//   __private long out_channel_num,
//   __private long out_height,
//   __private long out_width,
//   __private long stride,
//   __private long kernel_height,
//   __private long kernel_width
//   #ifdef USE_INSTANCE_NORM
//   ,
//   __global float* mean,
//   __global float* variance
//   #endif
//   )
// {
//   long out_height_idx  = get_global_id(0);
//   long out_width_idx = get_global_id(1);
//   long out_channel_idx = get_global_id(2);
//   long out_width_idx = width_workitem_id * UNROLL_CNT;

//   long in_channel_block_range = UPDIV4(in_channel_num);
//   long weight_offset = out_channel_block_idx * in_channel_block_range * kernel_height * kernel_width * 4;

//   float4 out0 = convert_float4(vload4(out_channel_block_idx, bias));
//   float4 out1 = out0;

//   long in_channel_block_idx = 0;
//   for (in_channel_block_idx = 0;
//        in_channel_block_idx < in_channel_block_range;
//        in_channel_block_idx++)
//   {
//     long in_hstart = out_height_idx * stride;
//     long in_channel_base_offset = in_channel_block_idx * in_height * in_width;
//     long in_height_idx = in_hstart;
//     for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
//     {
//       long line_offset_base = in_channel_base_offset + in_height_idx * in_width;
//       long in_wstart = out_width_idx * stride;
//       long in_width_idx = in_wstart;
//       for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
//       {
//         long in0_offset = line_offset_base + in_width_idx;
//         long in1_offset = in0_offset + stride;
//         float4 in0 = convert_float4(vload4(in0_offset, input));
//         float4 in1 = convert_float4(vload4(in1_offset, input));

//         float4 w0 = convert_float4(vload4(weight_offset, weight));
//         float4 w1 = convert_float4(vload4(weight_offset+1, weight));
//         float4 w2 = convert_float4(vload4(weight_offset+2, weight));
//         float4 w3 = convert_float4(vload4(weight_offset+3, weight));
//         weight_offset += 4;

//         out0 = mad(in0.x, w0, out0);
//         out0 = mad(in0.y, w1, out0);
//         out0 = mad(in0.z, w2, out0);
//         out0 = mad(in0.w, w3, out0);

//         out1 = mad(in1.x, w0, out1);
//         out1 = mad(in1.y, w1, out1);
//         out1 = mad(in1.z, w2, out1);
//         out1 = mad(in1.w, w3, out1);
//         in_width_idx += 1;
//       }
//       in_height_idx++;
//     }
//   }

// #ifdef USE_INSTANCE_NORM
//   float4 mean_val = convert_float4(vload4(out_channel_block_idx, mean));
//   float4 var_val = convert_float4(vload4(out_channel_block_idx, variance));
//   float4 var_coefficient = 1.0/sqrt(var_val+1e-5);
//   out0 = (out0 - mean_val) * var_coefficient;
//   out1 = (out1 - mean_val) * var_coefficient;
// #endif
// #ifdef USE_RELU
//   out0 = max(out0, 0.0);
//   out1 = max(out1, 0.0);
// #endif

// #ifdef USE_TANH
//   out0 = tanh(out0);
//   out1 = tanh(out1);
// #endif
//   long out0_offset = out_channel_block_idx * out_height * out_width +
//                     out_height_idx * out_width +
//                     out_width_idx;
//   long out1_offset = out0_offset + 1;
//   vstore4(out0, out0_offset, output);
//   vstore4(out1, out1_offset, output);
// }

#define UNROLL_CNT 2

__kernel void conv2d_nchw4_block(
  __global float* input, /* layout: CHW(4c) */
  __global float* weight, /* layout:  (C_out, C_in, KH, KW, 4c_o, 4c_in)*/
  __global float* bias, /* layout: (C, 4c) */
  __global float* output, /* layout: CHW(4c) */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  // each thread is responsible for 2*4 float output
  long out_channel_block_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long width_workitem_id = get_global_id(2);
  long out_width_idx = width_workitem_id * UNROLL_CNT;

  long in_channel_block_range = UPDIV4(in_channel_num);
  long weight_offset = out_channel_block_idx * in_channel_block_range * kernel_height * kernel_width * 4;

  float4 out0 = convert_float4(vload4(out_channel_block_idx, bias));
  float4 out1 = out0;

  for (long in_channel_block_idx = 0;
       in_channel_block_idx < in_channel_block_range;
       in_channel_block_idx++)
  {
    long in_hstart = out_height_idx * stride;
    long in_channel_base_offset = in_channel_block_idx * in_height * in_width;
    long in_height_idx = in_hstart;
    for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
    {
      long line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      long in_wstart = out_width_idx * stride;
      long in_width_idx = in_wstart;
      for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
      {
        long in0_offset = line_offset_base + in_width_idx;
        long in1_offset = in0_offset + stride;
        float4 in0 = convert_float4(vload4(in0_offset, input));
        float4 in1 = convert_float4(vload4(in1_offset, input));

        float4 w0 = convert_float4(vload4(weight_offset, weight));
        float4 w1 = convert_float4(vload4(weight_offset+1, weight));
        float4 w2 = convert_float4(vload4(weight_offset+2, weight));
        float4 w3 = convert_float4(vload4(weight_offset+3, weight));
        weight_offset += 4;

        out0 = mad(in0.x, w0, out0);
        out0 = mad(in0.y, w1, out0);
        out0 = mad(in0.z, w2, out0);
        out0 = mad(in0.w, w3, out0);

        out1 = mad(in1.x, w0, out1);
        out1 = mad(in1.y, w1, out1);
        out1 = mad(in1.z, w2, out1);
        out1 = mad(in1.w, w3, out1);
        in_width_idx += 1;
      }
      in_height_idx++;
    }
  }

#ifdef USE_INSTANCE_NORM
  float4 mean_val = convert_float4(vload4(out_channel_block_idx, mean));
  float4 var_val = convert_float4(vload4(out_channel_block_idx, variance));
  float4 var_coefficient = 1.0/sqrt(var_val+1e-5);
  out0 = (out0 - mean_val) * var_coefficient;
  out1 = (out1 - mean_val) * var_coefficient;
#endif
#ifdef USE_RELU
  out0 = max(out0, 0.0);
  out1 = max(out1, 0.0);
#endif

#ifdef USE_TANH
  out0 = tanh(out0);
  out1 = tanh(out1);
#endif
  long out0_offset = out_channel_block_idx * out_height * out_width +
                    out_height_idx * out_width +
                    out_width_idx;
  long out1_offset = out0_offset + 1;
  vstore4(out0, out0_offset, output);
  vstore4(out1, out1_offset, output);
}

__kernel void conv2d_nchw4_interleave_ver2(
  __global float* input, /* layout: CHW(4c) */
  __global float* weight, /* layout:  (C_out, C_in, KH, KW, 4c_o, 4c_in)*/
  __global float* bias, /* layout: (C, 4c) */
  __global float* output, /* layout: CHW(4c) */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  // each thread is responsible for 2*4 float output
  long out_channel_block_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long width_workitem_id = get_global_id(2);
  long out_width_idx = width_workitem_id * UNROLL_CNT;

  long in_channel_block_range = UPDIV4(in_channel_num);
  long weight_offset = out_channel_block_idx * in_channel_block_range * kernel_height * kernel_width * 4;

  float4 out0 = convert_float4(vload4(out_channel_block_idx, bias));
  float4 out1 = out0;
  float4 out2 = out0;
  float4 out3 = out0;

  for (long in_channel_block_idx = 0;
       in_channel_block_idx < in_channel_block_range;
       in_channel_block_idx++)
  {
    long in_hstart = out_height_idx * stride;
    long in_channel_base_offset = in_channel_block_idx * in_height * in_width;
    long in_height_idx = in_hstart;
    for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
    {
      long line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      long in_wstart = out_width_idx * stride;
      long in_width_idx = in_wstart;
      for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
      {
        long in0_offset = line_offset_base + in_width_idx;
        long in1_offset = in0_offset + stride * 4;
        float4 in0 = convert_float4(vload4(in0_offset, input));
        float4 in1 = convert_float4(vload4(in1_offset, input));
        float4 in2 = convert_float4(vload4(in0_offset + stride * 4 * 2, input));
        float4 in3 = convert_float4(vload4(in0_offset + stride * 4 * 3, input));

        float4 w0 = convert_float4(vload4(weight_offset, weight));
        float4 w1 = convert_float4(vload4(weight_offset+1, weight));
        float4 w2 = convert_float4(vload4(weight_offset+2, weight));
        float4 w3 = convert_float4(vload4(weight_offset+3, weight));
        weight_offset += 4;

        out0 = mad(in0.x, w0, out0);
        out0 = mad(in0.y, w1, out0);
        out0 = mad(in0.z, w2, out0);
        out0 = mad(in0.w, w3, out0);

        out1 = mad(in1.x, w0, out1);
        out1 = mad(in1.y, w1, out1);
        out1 = mad(in1.z, w2, out1);
        out1 = mad(in1.w, w3, out1);

        out2 = mad(in2.x, w0, out2);
        out2 = mad(in2.y, w1, out2);
        out2 = mad(in2.z, w2, out2);
        out2 = mad(in2.w, w3, out2);

        out3 = mad(in3.x, w0, out3);
        out3 = mad(in3.y, w1, out3);
        out3 = mad(in3.z, w2, out3);
        out3 = mad(in3.w, w3, out3);
        in_width_idx += 1;
      }
      in_height_idx++;
    }
  }

  long out0_offset = out_channel_block_idx * out_height * out_width +
                    out_height_idx * out_width +
                    out_width_idx;
  long out1_offset = out0_offset + 4;
  vstore4(out0, out0_offset, output);
  vstore4(out1, out1_offset, output);
  vstore4(out1, out0_offset + 4 * 2, output);
  vstore4(out1, out0_offset + 4 * 3, output);
}

__kernel void conv2d_nchw4_interleave(
  __global float* input, /* layout: CHW(4c) */
  __global float* weight, /* layout:  (C_out, C_in, KH, KW, 4c_o, 4c_in)*/
  __global float* bias, /* layout: (C, 4c) */
  __global float* output, /* layout: CHW(4c) */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  // each thread is responsible for 2*4 float output
  long out_channel_block_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long width_workitem_id = get_global_id(2);
  long out_width_idx = width_workitem_id * UNROLL_CNT;

  long in_channel_block_range = UPDIV4(in_channel_num);
  long weight_offset = out_channel_block_idx * in_channel_block_range * kernel_height * kernel_width * 4;

  float4 out0 = convert_float4(vload4(out_channel_block_idx, bias));
  float4 out1 = out0;

  for (long in_channel_block_idx = 0;
       in_channel_block_idx < in_channel_block_range;
       in_channel_block_idx++)
  {
    long in_hstart = out_height_idx * stride;
    long in_channel_base_offset = in_channel_block_idx * in_height * in_width;
    long in_height_idx = in_hstart;
    for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
    {
      long line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      long in_wstart = out_width_idx * stride;
      long in_width_idx = in_wstart;
      for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
      {
        long in0_offset = line_offset_base + in_width_idx;
        long in1_offset = in0_offset + stride * 4;
        float4 in0 = convert_float4(vload4(in0_offset, input));
        float4 in1 = convert_float4(vload4(in1_offset, input));

        float4 w0 = convert_float4(vload4(weight_offset, weight));
        float4 w1 = convert_float4(vload4(weight_offset+1, weight));
        float4 w2 = convert_float4(vload4(weight_offset+2, weight));
        float4 w3 = convert_float4(vload4(weight_offset+3, weight));
        weight_offset += 4;

        out0 = mad(in0.x, w0, out0);
        out0 = mad(in0.y, w1, out0);
        out0 = mad(in0.z, w2, out0);
        out0 = mad(in0.w, w3, out0);

        out1 = mad(in1.x, w0, out1);
        out1 = mad(in1.y, w1, out1);
        out1 = mad(in1.z, w2, out1);
        out1 = mad(in1.w, w3, out1);
        in_width_idx += 1;
      }
      in_height_idx++;
    }
  }

#ifdef USE_INSTANCE_NORM
  float4 mean_val = convert_float4(vload4(out_channel_block_idx, mean));
  float4 var_val = convert_float4(vload4(out_channel_block_idx, variance));
  float4 var_coefficient = 1.0/sqrt(var_val+1e-5);
  out0 = (out0 - mean_val) * var_coefficient;
  out1 = (out1 - mean_val) * var_coefficient;
#endif
#ifdef USE_RELU
  out0 = max(out0, 0.0);
  out1 = max(out1, 0.0);
#endif

#ifdef USE_TANH
  out0 = tanh(out0);
  out1 = tanh(out1);
#endif
  long out0_offset = out_channel_block_idx * out_height * out_width +
                    out_height_idx * out_width +
                    out_width_idx;
  long out1_offset = out0_offset + 4;
  vstore4(out0, out0_offset, output);
  vstore4(out1, out1_offset, output);
}


__kernel void conv2d_nhwc(
  __global float* input, /* layout: HWC */
  __global float* weight, /* layout: HWOI */
  __global float* bias,
  __global float* output, /* layout: HWC */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  long out_height_idx  = get_global_id(0);
  long out_width_idx = get_global_id(1);
  long out_channel_idx = get_global_id(2);

  long weight_offset = out_channel_idx * in_channel_num * kernel_height * kernel_width;

  float out0 = bias[out_channel_idx];

  long input_upper_left_offset = out_height_idx * in_width * in_channel_num + out_width_idx * in_channel_num;
  for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
  {
    long input_line_offset = input_upper_left_offset + kh_idx * in_width * in_channel_num;
    for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
    {
      long input_wid_offset = input_line_offset + kw_idx * in_channel_num;
      for (long in_channel_idx = 0;
          in_channel_idx < in_channel_num;
          in_channel_idx++)
      {
        long in0_offset = input_wid_offset + in_channel_idx;
        float in0 = input[in0_offset];
        float w = weight[weight_offset];
        weight_offset++;
        out0 = mad(in0, w, out0);
      }
    }
  }
  long out0_offset = out_height_idx * out_width * out_channel_num +
                    out_width_idx * out_channel_num +
                    out_channel_idx;
  output[out0_offset] = out0;
}

__kernel void conv2d_nhwc_ver2(
  __global float* input, /* layout: HWC */
  __global float* weight, /* layout: HWOI */
  __global float* bias,
  __global float* output, /* layout: HWC */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  long out_height_idx = get_global_id(0);
  long width_workitem_id = get_global_id(1);
  long out_channel_block_idx = get_global_id(2);
  long out_width_idx = width_workitem_id * UNROLL_CNT;

  long in_channel_block_range = UPDIV4(in_channel_num);
  long out_channel_block_range = (out_channel_num + 3) / 4;

  float4 out0 = convert_float4(vload4(out_channel_block_idx, bias));
  float4 out1 = out0;

  long input_upper_left_offset = out_height_idx * stride * in_width * in_channel_block_range + out_width_idx * stride * in_channel_block_range;
  for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
  {
    long input_line_offset = input_upper_left_offset + kh_idx * in_width * in_channel_block_range;
    for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
    {
      long input_wid_offset = input_line_offset + kw_idx * in_channel_block_range;
      for (long in_channel_block_idx = 0;
          in_channel_block_idx < in_channel_block_range;
          in_channel_block_idx++)
      {
        long weight_offset = kh_idx * kernel_width * out_channel_block_range * 4 * in_channel_block_range
                             + kw_idx * out_channel_block_range * 4 * in_channel_block_range
                             + out_channel_block_idx * 4 * in_channel_block_range
                             + in_channel_block_idx;
        long in0_offset = input_wid_offset + in_channel_block_idx;
        long in1_offset = in0_offset + stride * in_channel_block_range;
        float4 in0 = convert_float4(vload4(in0_offset, input));
        float4 in1 = convert_float4(vload4(in1_offset, input));

        float4 w0 = convert_float4(vload4(weight_offset, weight));
        float4 w1 = convert_float4(vload4(weight_offset+in_channel_block_range, weight));
        float4 w2 = convert_float4(vload4(weight_offset+2 * in_channel_block_range, weight));
        float4 w3 = convert_float4(vload4(weight_offset+3 * in_channel_block_range, weight));

        out0.x += dot(in0, w0);
        out0.y += dot(in0, w1);
        out0.z += dot(in0, w2);
        out0.w += dot(in0, w3);
        out1.x += dot(in1, w0);
        out1.y += dot(in1, w1);
        out1.z += dot(in1, w2);
        out1.w += dot(in1, w3);
      }
    }
  }
  long out0_offset = out_height_idx * out_width * out_channel_block_range +
                     out_width_idx * out_channel_block_range +
                     out_channel_block_idx;
  long out1_offset = out0_offset + out_channel_block_range;
  vstore4(out0, out0_offset, output);
  vstore4(out1, out1_offset, output);
}

__kernel void conv2d_nhwc_ver3(
  __global float* input, /* layout: HWC */
  __global float* weight, /* layout: HWOI */
  __global float* bias,
  __global float* output, /* layout: HWC */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  long out_height_idx = get_global_id(0);
  long width_workitem_id = get_global_id(1);
  long out_channel_block_idx = get_global_id(2);
  long out_width_idx = width_workitem_id * UNROLL_CNT;

  long in_channel_block_range = UPDIV4(in_channel_num);
  long out_channel_block_range = (out_channel_num + 3) / 4;

  float4 out0 = convert_float4(vload4(out_channel_block_idx, bias));
  float4 out1 = out0;
  float4 out2 = out0;
  float4 out3 = out0;

  long input_upper_left_offset = out_height_idx * stride * in_width * in_channel_block_range + out_width_idx * stride * in_channel_block_range;
  for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
  {
    long input_line_offset = input_upper_left_offset + kh_idx * in_width * in_channel_block_range;
    for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
    {
      long input_wid_offset = input_line_offset + kw_idx * in_channel_block_range;
      for (long in_channel_block_idx = 0;
          in_channel_block_idx < in_channel_block_range;
          in_channel_block_idx++)
      {
        long weight_offset = kh_idx * kernel_width * out_channel_block_range * 4 * in_channel_block_range
                             + kw_idx * out_channel_block_range * 4 * in_channel_block_range
                             + out_channel_block_idx * 4 * in_channel_block_range
                             + in_channel_block_idx;
        long in0_offset = input_wid_offset + in_channel_block_idx;
        long in1_offset = in0_offset + stride * in_channel_block_range;
        float4 in0 = convert_float4(vload4(in0_offset, input));
        float4 in1 = convert_float4(vload4(in1_offset, input));
        float4 in2 = convert_float4(vload4(in0_offset + stride * in_channel_block_range * 2, input));
        float4 in3 = convert_float4(vload4(in0_offset + stride * in_channel_block_range * 3, input));


        float4 w0 = convert_float4(vload4(weight_offset, weight));
        float4 w1 = convert_float4(vload4(weight_offset+in_channel_block_range, weight));
        float4 w2 = convert_float4(vload4(weight_offset+2 * in_channel_block_range, weight));
        float4 w3 = convert_float4(vload4(weight_offset+3 * in_channel_block_range, weight));

        out0.x += dot(in0, w0);
        out0.y += dot(in0, w1);
        out0.z += dot(in0, w2);
        out0.w += dot(in0, w3);
        out1.x += dot(in1, w0);
        out1.y += dot(in1, w1);
        out1.z += dot(in1, w2);
        out1.w += dot(in1, w3);
        out2.x += dot(in2, w0);
        out2.y += dot(in2, w1);
        out2.z += dot(in2, w2);
        out2.w += dot(in2, w3);
        out3.x += dot(in3, w0);
        out3.y += dot(in3, w1);
        out3.z += dot(in3, w2);
        out3.w += dot(in3, w3);
      }
    }
  }
  long out0_offset = out_height_idx * out_width * out_channel_block_range +
                     out_width_idx * out_channel_block_range +
                     out_channel_block_idx;
  long out1_offset = out0_offset + out_channel_block_range;
  vstore4(out0, out0_offset, output);
  vstore4(out1, out1_offset, output);
  vstore4(out0, out0_offset + out_channel_block_range * 2, output);
  vstore4(out0, out0_offset + out_channel_block_range * 3, output);
}


__kernel void conv2d_nchw(
  __global float* input, /* layout: CHW */
  __global float* weight, /* layout: OIHW */
  __global float* bias,
  __global float* output, /* layout: CHW */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  long out_channel_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long out_width_idx = get_global_id(2);

  long weight_offset = out_channel_idx * in_channel_num * kernel_height * kernel_width;

  float out0 = bias[out_channel_idx];

  for (long in_channel_idx = 0;
       in_channel_idx < in_channel_num;
       in_channel_idx++)
  {
    long in_hstart = out_height_idx * stride;
    long in_channel_base_offset = in_channel_idx * in_height * in_width;
    long in_height_idx = in_hstart;
    for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
    {
      long line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      long in_wstart = out_width_idx * stride;
      long in_width_idx = in_wstart;
      for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
      {
        long in0_offset = line_offset_base + in_width_idx;
        float in0 = input[in0_offset];

        float w = weight[weight_offset];
        weight_offset ++;
        out0 = mad(in0, w, out0);
        in_width_idx += 1;
      }
      in_height_idx++;
    }
  }
  long out0_offset = out_channel_idx * out_height * out_width +
                    out_height_idx * out_width +
                    out_width_idx;
  output[out0_offset] = out0;
}


__kernel void conv2d_nchw_ver2(
  __global float* input, /* layout: CHW */
  __global float* weight, /* layout: OIHW */
  __global float* bias,
  __global float* output, /* layout: CHW */
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long out_channel_num,
  __private long out_height,
  __private long out_width,
  __private long stride,
  __private long kernel_height,
  __private long kernel_width
  #ifdef USE_INSTANCE_NORM
  ,
  __global float* mean,
  __global float* variance
  #endif
  )
{
  long out_channel_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long out_width_workitem_idx = get_global_id(2);
  long out_width_idx = out_width_workitem_idx * 4;

  long weight_offset = out_channel_idx * in_channel_num * kernel_height * kernel_width;

  float out0 = bias[out_channel_idx];
  float out1 = out0;
  float out2 = out0;
  float out3 = out0;
  float out4 = out0;
  float out5 = out0;
  float out6 = out0;
  float out7 = out0;


  for (long in_channel_idx = 0;
       in_channel_idx < in_channel_num;
       in_channel_idx++)
  {
    long in_hstart = out_height_idx * stride;
    long in_channel_base_offset = in_channel_idx * in_height * in_width;
    long in_height_idx = in_hstart;
    for (long kh_idx = 0; kh_idx < kernel_height; kh_idx++)
    {
      long line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      long in_wstart = out_width_idx * stride;
      long in_width_idx = in_wstart;
      for (long kw_idx = 0; kw_idx < kernel_width; kw_idx++)
      {
        long in0_offset = line_offset_base + in_width_idx;
        float in0 = input[in0_offset];
        float in1 = input[in0_offset+1];
        float in2 = input[in0_offset+2];
        float in3 = input[in0_offset+3];
        float in4 = input[in0_offset+4];
        float in5 = input[in0_offset+5];
        float in6 = input[in0_offset+6];
        float in7 = input[in0_offset+7];

        float w = weight[weight_offset];
        weight_offset ++;
        out0 = mad(in0, w, out0);
        out1 = mad(in1, w, out1);
        out2 = mad(in2, w, out2);
        out3 = mad(in3, w, out3);
        out4 = mad(in4, w, out4);
        out5 = mad(in5, w, out5);
        out6 = mad(in6, w, out6);
        out7 = mad(in7, w, out7);
        in_width_idx += 1;
      }
      in_height_idx++;
    }
  }
  long out0_offset = out_channel_idx * out_height * out_width +
                    out_height_idx * out_width +
                    out_width_idx;
  output[out0_offset] = out0;
  output[out0_offset+1] = out1;
  output[out0_offset+2] = out2;
  output[out0_offset+3] = out3;
  output[out0_offset+4] = out4;
  output[out0_offset+5] = out5;
  output[out0_offset+6] = out6;
  output[out0_offset+7] = out7;
}
