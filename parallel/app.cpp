#include "parallel/app.hpp";

namespace parallel {

int run(cl_mem input, cl_mem output, gan_buffer_t &buf, model &cycleGAN,
        cl_command_queue queue) {
  reflectpad_2d(queue, input, buf.o_0, 4/4, 256, 256, 3);
  conv2d_exec_async(queue, buf.o_0, cycleGAN.m_1.weight_buf,
                    cycleGAN.m_1.bias_buf, buf.o_1, cycleGAN.m_2.mean_buf,
                    cycleGAN.m_2.variance_buf, 4/4, 262, 262, 64/4,
                    256, 256,1, 7, 7, true,
                    activation::RELU);

  zeropad_2d_onepix(queue, buf.o_1, buf.o_1_p, 64/4, 256, 256);
  conv2d_exec_async(queue, buf.o_1_p, cycleGAN.m_4.weight_buf,
                  cycleGAN.m_4.bias_buf, buf.o_2, cycleGAN.m_5.mean_buf,
                  cycleGAN.m_5.variance_buf, 64/4, 258, 258, 128/4,
                  128, 128,2, 3, 3, true,
                  activation::RELU);

  zeropad_2d_onepix(queue, buf.o_2, buf.o_2_p, 128/4, 128, 128);
  conv2d_exec_async(queue, buf.o_2_p, cycleGAN.m_7.weight_buf,
                    cycleGAN.m_7.bias_buf, buf.o_3, cycleGAN.m_8.mean_buf,
                    cycleGAN.m_8.variance_buf, 128/4, 130, 130, 256/4,
                    64, 64,2, 3, 3, true,
                    activation::RELU);

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

  // conv transpose
  conv2d_transpose_3x3_stride2_norm_relu_exec_async(queue, buf.r8_3, cycleGAN.m_19.weight_buf, cycleGAN.m_19.bias_buf,
      buf.o_4, cycleGAN.m_20.mean_buf, cycleGAN.m_20.variance_buf, 256/4, 64, 64,
      128/4, 128, 128, true, activation::RELU);

  conv2d_transpose_3x3_stride2_norm_relu_exec_async(queue, buf.o_4, cycleGAN.m_22.weight_buf, cycleGAN.m_22.bias_buf,
      buf.o_5, cycleGAN.m_23.mean_buf, cycleGAN.m_23.variance_buf, 128/4, 128, 128,
      64/4, 256, 256, true, activation::RELU);

  // conv
  reflectpad_2d(queue, buf.o_5, buf.o_6, 64/4, 256, 256, 3);
  conv2d_exec_async(queue, buf.o_6, cycleGAN.m_26.weight_buf, cycleGAN.m_26.bias_buf, output, nullptr,
                    nullptr, 64/4, 262, 262, 4/4,
                    256, 256,1, 7, 7, false,
                    activation::TANH);
}

void init_gan_buffer(gan_buffer_t &buf, cl_context context) {
    buf.o_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 4 * 262 * 262, NULL, NULL);
    buf.o_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 64 * 256 * 256, NULL, NULL);
    buf.o_1_p = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 64 * 258 * 258, NULL, NULL);
    buf.o_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 128 * 128 * 128, NULL, NULL);
    buf.o_2_p = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 128 * 130 * 130, NULL, NULL);
    buf.o_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64, NULL, NULL);
    buf.r0_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r0_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r0_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r0_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL); 
    buf.r1_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r1_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r1_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r1_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);     
    buf.r2_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r2_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r2_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r3_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r3_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r3_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r3_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL); 
    buf.r4_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r4_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r4_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r4_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r5_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r5_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r5_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r5_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);  
    buf.r6_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r6_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r6_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r6_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL); 
    buf.r7_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r7_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r7_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r7_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);     
    buf.r8_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r8_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL);
    buf.r8_2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 66 * 66,  NULL, NULL);
    buf.r8_3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 256 * 64 * 64,  NULL, NULL); 
    buf.o_4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 128 * 128 * 128, NULL, NULL);   
    buf.o_5 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 64 * 256 * 256, NULL, NULL);     
    buf.o_6 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * 64 * 262 * 262, NULL, NULL);
}

void release_gan_buffer(gan_buffer_t &buf) {
    clReleaseMemObject(buf.o_0);
    clReleaseMemObject(buf.o_1);
    clReleaseMemObject(buf.o_1_p);
    clReleaseMemObject(buf.o_2);
    clReleaseMemObject(buf.o_2_p);
    clReleaseMemObject(buf.o_3);
    clReleaseMemObject(buf.r0_0);
    clReleaseMemObject(buf.r0_1);
    clReleaseMemObject(buf.r0_2);
    clReleaseMemObject(buf.r0_3);
    clReleaseMemObject(buf.r1_0);
    clReleaseMemObject(buf.r1_1);
    clReleaseMemObject(buf.r1_2);
    clReleaseMemObject(buf.r1_3);
    clReleaseMemObject(buf.r2_0);
    clReleaseMemObject(buf.r2_1);
    clReleaseMemObject(buf.r2_2);
    clReleaseMemObject(buf.r2_3);
    clReleaseMemObject(buf.r3_0);
    clReleaseMemObject(buf.r3_1);
    clReleaseMemObject(buf.r3_2);
    clReleaseMemObject(buf.r3_3);
    clReleaseMemObject(buf.r4_0);
    clReleaseMemObject(buf.r4_1);
    clReleaseMemObject(buf.r4_2);
    clReleaseMemObject(buf.r4_3);
    clReleaseMemObject(buf.r5_0);
    clReleaseMemObject(buf.r5_1);
    clReleaseMemObject(buf.r5_2);
    clReleaseMemObject(buf.r5_3);
    clReleaseMemObject(buf.r6_0);
    clReleaseMemObject(buf.r6_1);
    clReleaseMemObject(buf.r6_2);
    clReleaseMemObject(buf.r6_3);
    clReleaseMemObject(buf.r7_0);
    clReleaseMemObject(buf.r7_1);
    clReleaseMemObject(buf.r7_2);
    clReleaseMemObject(buf.r7_3);
    clReleaseMemObject(buf.r8_0);
    clReleaseMemObject(buf.r8_1);
    clReleaseMemObject(buf.r8_2);
    clReleaseMemObject(buf.r8_3);
    clReleaseMemObject(buf.o_4);
    clReleaseMemObject(buf.o_5);
    clReleaseMemObject(buf.o_6);
}

}