#include "common.h"

#define UNROLL_CNT 2

__kernel void conv2d(
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
