__kernel void zero_pad_2d_onepix(
  __global const float *input,
  __global float *output,
  __private long in_channel_num,
  __private long in_height,
  __private long in_width
) {
  long out_channel_block_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long width_workitem_id = get_global_id(2);
  long out_width_idx = width_workitem_id * 2;

  float4 out0, out1;
  long in_offset = out_channel_block_idx * in_height * in_width + (out_height_idx - 1) * in_width + (out_width_idx - 1);
  long out_offset = out_channel_block_idx * (in_height + 2) * (in_width + 2) + out_height_idx * (in_width + 2) + out_width_idx;
  if (out_height_idx > 0 && out_height_idx <= in_height
      && out_width_idx > 0 && out_width_idx < in_width)
  {
    out0 = convert_float4(vload4(in_offset, input));
    out1 = convert_float4(vload4(in_offset + 1, input));
  } else if (out_height_idx == 0 || out_height_idx == in_height + 1)
  {
    out0 = 0;
    out1 = 0;
  } else if (out_width_idx == 0) {
    out0 = 0;
    out1 = convert_float4(vload4(in_offset + 1, input));
  } else {
    // assert(out_width_idx == 0 || out_width_idx == in_width);
    out0 = convert_float4(vload4(in_offset, input));
    out1 = 0;
  }
  vstore4(out0, out_offset, output);
  vstore4(out1, out_offset+1, output);
}

__kernel void reflect_pad_2d_onepix(
  __global const float *input,
  __global float *output,
  __private long in_channel_num,
  __private long in_height,
  __private long in_width
) {
  long out_channel_block_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long width_workitem_id = get_global_id(2);
  long out_width_idx = width_workitem_id * 2;

  float4 out0, out1;
  long in0_offset = out_channel_block_idx * in_height * in_width + (out_height_idx - 1) * in_width + (out_width_idx - 1);
  long in1_offset = in0_offset + 1;
  if (out_height_idx > 0 && out_height_idx <= in_height
      && out_width_idx > 0 && out_width_idx < in_width)
  {
    // do nothing
  } else
  {
    if (out_height_idx == 0)
    {
      in0_offset += 2 * in_width;
      in1_offset += 2 * in_width;
    } else if (out_height_idx == in_height + 1)
    {
      in0_offset -= 2 * in_width;
      in1_offset -= 2 * in_width;
    }
    if (out_width_idx == 0) {
      in0_offset += 2;
    } else if (out_width_idx == in_width)
    {
      in1_offset -= 2;
    }
  }
  long out_offset = out_channel_block_idx * (in_height + 2) * (in_width + 2) + out_height_idx * (in_width + 2) + out_width_idx;
  out0 = convert_float4(vload4(in0_offset, input));
  out1 = convert_float4(vload4(in1_offset, input));
  vstore4(out0, out_offset, output);
  vstore4(out1, out_offset+1, output);
}

__kernel void reflect_pad_2d(
  __global const float *input,
  __global float *output,
  __private long in_channel_num,
  __private long in_height,
  __private long in_width,
  __private long padding
) {
  long out_channel_block_idx  = get_global_id(0);
  long out_height_idx = get_global_id(1);
  long out_width_idx = get_global_id(2);

  float4 out0;
  long in0_offset = out_channel_block_idx * in_height * in_width + (out_height_idx - 1) * in_width + (out_width_idx - 1);
  long in1_offset = in0_offset + 1;
  if (out_height_idx >= padding && out_height_idx < in_height + padding
      && out_width_idx >= padding && out_width_idx < in_width + padding)
  {
    // do nothing
  } else
  {
    if (out_height_idx < padding)
    {
      in0_offset += 2 * (padding - out_height_idx) * in_width;
    } else if (out_height_idx >= in_height + padding)
    {
      in0_offset -= 2 * (in_height + padding + 1 - out_height_idx) * in_width;
    }
    if (out_width_idx < padding) {
      in0_offset += 2 * (padding - out_width_idx);
    } else if (out_width_idx >= in_width + padding)
    {
      in0_offset -= 2 * (in_width + padding + 1 - out_width_idx);
    }
  }
  long out_offset = out_channel_block_idx * (in_height + 2) * (in_width + 2) + out_height_idx * (in_width + 2) + out_width_idx;
  out0 = convert_float4(vload4(in0_offset, input));
  vstore4(out0, out_offset, output);
}