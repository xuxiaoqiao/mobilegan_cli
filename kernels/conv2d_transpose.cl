#include "common.h"

#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3


// ASSUMPTION: STRIDE=2, PADDING=1, OUT_PADDING=1

__kernel void conv2d_transpose_3x3_stride2(
  __global float* input, /* layout: CHW(4c) */
  __global float* weight, /* layout:  (C_out, C_in, KH, KW, 4c_in, 4c_o)*/
  __global float* bias, /* layout: (C, 4c) */
  __global float* output, /* layout: CHW(4c) */
  __private int in_channel_num,
  __private int in_height,
  __private int in_width,
  __private int out_channel_num,
  __private int out_height,
  __private int out_width,
  __global float* mean,
  __global float* variance
  )
{
  int out_channel_block_idx  = get_global_id(0);
  int out_height_idx = get_global_id(1);
  int width_quad_idx = get_global_id(2);
  int out_width_idx = width_quad_idx >> 2;
  int quad_idx = width_quad_idx & 3;

  int in_channel_block_range = UPDIV4(in_channel_num);
  int out_channel_idx = out_channel_block_idx * 4 + quad_idx;
  int bias_offset = out_channel_idx;
  float out0 = bias[bias_offset];
  if ((out_height_idx % 2) == 0 && (out_width_idx % 2) == 0) {
    for (int in_channel_block_idx = 0;
        in_channel_block_idx < in_channel_block_range;
        in_channel_block_idx++)
    {
      int weight_offset = out_channel_block_idx * in_channel_block_range * KERNEL_HEIGHT * KERNEL_WIDTH * 4
                          + 1 * KERNEL_WIDTH * 4
                          + 4
                          + quad_idx;
      int in_channel_base_offset = in_channel_block_idx * in_height * in_width;
      int in_height_idx = out_height_idx >> 1;

      int line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      int in_width_idx = out_width_idx >> 1;

      int in_offset = line_offset_base + in_width_idx;
      float4 in0 = convert_float4(vload4(in_offset, input));
      float4 w0 = convert_float4(vload4(weight_offset, weight));
      out0 += dot(in0, w0);
    }
  } else if ((out_height_idx % 2) != 0 && (out_width_idx % 2) == 0) {
    for (int in_channel_block_idx = 0;
        in_channel_block_idx < in_channel_block_range;
        in_channel_block_idx++)
    {
      // UPPER-MIDDLE input
      int weight_offset = out_channel_block_idx * in_channel_block_range * KERNEL_HEIGHT * KERNEL_WIDTH * 4
                          + 2 * KERNEL_WIDTH * 4
                          + 4
                          + quad_idx;
      int in_channel_base_offset = in_channel_block_idx * in_height * in_width;
      int in_height_idx = out_height_idx >> 1;

      int line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      int in_width_idx = out_width_idx >> 1;

      int in_offset = line_offset_base + in_width_idx;
      float4 in0 = convert_float4(vload4(in_offset, input));
      float4 w0 = convert_float4(vload4(weight_offset, weight));
      out0 += dot(in0, w0);

      if (out_height_idx + 1 != out_height) {
        // LOWER-MIDDLE input
        in0 = convert_float4(vload4(in_offset + in_width, input));
        w0 = convert_float4(vload4(weight_offset - 2 * KERNEL_WIDTH * 4, weight));
        out0 += dot(in0, w0);
      }
    }
  } else if ((out_height_idx % 2) == 0 && (out_width_idx % 2) != 0) {
    for (int in_channel_block_idx = 0;
      in_channel_block_idx < in_channel_block_range;
      in_channel_block_idx++)
    {
      // MID-LEFT input
      int weight_offset = out_channel_block_idx * in_channel_block_range * KERNEL_HEIGHT * KERNEL_WIDTH * 4
                          + 1 * KERNEL_WIDTH * 4
                          + 2 * 4
                          + quad_idx;
      int in_channel_base_offset = in_channel_block_idx * in_height * in_width;
      int in_height_idx = out_height_idx >> 1;

      int line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      int in_width_idx = out_width_idx >> 1;

      int in_offset = line_offset_base + in_width_idx;
      float4 in0 = convert_float4(vload4(in_offset, input));
      float4 w0 = convert_float4(vload4(weight_offset, weight));
      out0 += dot(in0, w0);

      if (out_width_idx + 1 != out_width) {
        // MID-RIGHT input
        in0 = convert_float4(vload4(in_offset + 1, input));
        w0 = convert_float4(vload4(weight_offset - 2 * 4, weight));
        out0 += dot(in0, w0);
      }
    }
  } else {
    for (int in_channel_block_idx = 0;
      in_channel_block_idx < in_channel_block_range;
      in_channel_block_idx++)
    {
      // UPPER-LEFT input
      int weight_offset = out_channel_block_idx * in_channel_block_range * KERNEL_HEIGHT * KERNEL_WIDTH * 4
                          + 2 * KERNEL_WIDTH * 4
                          + 2 * 4
                          + quad_idx;
      int in_channel_base_offset = in_channel_block_idx * in_height * in_width;
      int in_height_idx = out_height_idx >> 1;

      int line_offset_base = in_channel_base_offset + in_height_idx * in_width;
      int in_width_idx = out_width_idx >> 1;

      int in_offset = line_offset_base + in_width_idx;
      float4 in0 = convert_float4(vload4(in_offset, input));
      float4 w0 = convert_float4(vload4(weight_offset, weight));
      out0 += dot(in0, w0);

      if (out_width_idx + 1 != out_width) {
        // UPPER_RIGHT input
        in0 = convert_float4(vload4(in_offset + 1, input));
        w0 = convert_float4(vload4(weight_offset - 2 * 4, weight));
        out0 += dot(in0, w0);
      }
      if (out_height_idx + 1 != out_height) {
        // LOWER_LEFT input
        in0 = convert_float4(vload4(in_offset + in_width, input));
        w0 = convert_float4(vload4(weight_offset - 2 * KERNEL_WIDTH * 4, weight));
        out0 += dot(in0, w0);
        if (out_width_idx + 1 != out_width) {
          // LOWER_RIGHT input
          in0 = convert_float4(vload4(in_offset + 1 + in_width, input));
          w0 = convert_float4(vload4(weight_offset - 2 * 4 - 2 * KERNEL_WIDTH * 4, weight));
          out0 += dot(in0, w0);
        }
      }
    }
  }

#ifdef USE_INSTANCE_NORM
  float mean_val = mean[out_channel_idx];
  float var_val = variance[out_channel_idx];
  out0 = (out0 - mean_val) / sqrt(var_val+1e-5);
#endif

#ifdef USE_RELU
  out0 = max(out0, 0.0);
#endif

  int out0_offset = out_channel_block_idx * out_height * out_width * 4 +
                    out_height_idx * out_width * 4 +
                    out_width_idx * 4 + quad_idx;
  output[out0_offset] = out0;
}
