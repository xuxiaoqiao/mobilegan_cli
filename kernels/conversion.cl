// precondition: assert(channel_num == 3 || channel_num mod 4 == 0)

__kernel void conv_chw_to_chw4(
  __global float* input,
  __global float* output,
  __private int channel_num,
  __private int height,
  __private int width
)
{
  int out_channel_block_idx = get_global_id(0);
  int out_height_idx = get_global_id(1);
  int out_width_idx = get_global_id(2);

  int height_width_product = height * width;
  int in0_offset = out_channel_block_idx * 4 * height_width_product + out_height_idx * width + out_width_idx;
  int in1_offset = in0_offset + height_width_product;
  int in2_offset = in1_offset + height_width_product;
  float in0 = convert_float(vload(in0_offset, input));
  float in1 = convert_float(vload(in1_offset, input));
  float in2 = convert_float(vload(in2_offset, input));
  float in3;
  if (channel_num == 3) {
    in3 = 0;
  } else {
    int in3_offset = in2_offset + height_width_product;
    in3 = convert_float(vload(in3_offset, input));
  }

  float4 out = {in0, in1, in2, in3};
  int out_offset = out_channel_block_idx * height * width
                   + out_height_idx * width + out_width_idx;
  vstore4(out, out_offset, output);
}

__kernel void conv_chw4_to_chw(
  __global float* input,
  __global float* output,
  __private int channel_num,
  __private int height,
  __private int width
)
{
  int out_channel_idx = get_global_id(0);
  int out_height_idx = get_global_id(1);
  int out_width_idx = get_global_id(2);
  int in_channel_blk_idx = (out_channel_idx + 3) >> 2;
  int in_channel_quad_idx = (out_channel_idx & 4);
  int in_offset = (in_channel_blk_idx * height * width + out_height_idx * width + out_width_idx) * 4 + in_quad_idx;
  int out_offset = out_channel_idx * height * width + out_height_idx * width + out_width_idx;
  float val;
  val = convert_float(vload(in_offset, input));
  vstore4(out, out_offset, output);
}