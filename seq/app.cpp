#include "seq/app.hpp"
#include <ctime>
using namespace std;

int run(const Tensor3D &input, Tensor3D &output, gan_buffer_t &buf,
        model &cycleGAN) {
  clock_t begin = clock();
  double timeSeconds;
  clock_t timestamp = clock();

  reflection_pad_2d(input, buf.o_0, 3);
  conv2d(buf.o_0, cycleGAN.m_1.weight, cycleGAN.m_1.bias, buf.o_1, 3, 262, 262, 64, 1, 7, 7);
  instance_norm(buf.o_1, cycleGAN.m_2.mean, cycleGAN.m_2.variance);
  relu(buf.o_1);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for conv2d\n", timeSeconds);
  timestamp = clock();
  
  zero_pad_2d(buf.o_1, buf.o_1_p, 1);
  conv2d(buf.o_1_p, cycleGAN.m_4.weight, cycleGAN.m_4.bias, buf.o_2, 64, 258, 258, 128, 2, 3, 3);
  instance_norm(buf.o_2, cycleGAN.m_5.mean, cycleGAN.m_5.variance);
  relu(buf.o_2);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for conv2d\n", timeSeconds);
  timestamp = clock();

  zero_pad_2d(buf.o_2, buf.o_2_p, 1);
  conv2d(buf.o_2_p, cycleGAN.m_7.weight, cycleGAN.m_7.bias, buf.o_3, 128, 130, 130, 256, 2, 3, 3);
  instance_norm(buf.o_3, cycleGAN.m_8.mean, cycleGAN.m_8.variance);
  relu(buf.o_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for conv2d\n", timeSeconds);
  timestamp = clock();

  // resnet block 0
  reflection_pad_2d(buf.o_3, buf.r0_0, 1);
  conv2d(buf.r0_0, cycleGAN.m_10.r_1.weight, cycleGAN.m_10.r_1.bias, buf.r0_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r0_1, cycleGAN.m_10.r_2.mean, cycleGAN.m_10.r_2.variance);
  relu(buf.r0_1);
  reflection_pad_2d(buf.r0_1, buf.r0_2, 1);
  conv2d(buf.r0_2, cycleGAN.m_10.r_5.weight, cycleGAN.m_10.r_5.bias, buf.r0_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r0_3, cycleGAN.m_10.r_6.mean, cycleGAN.m_10.r_6.variance);
  add(buf.o_3, buf.r0_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 0\n", timeSeconds);
  timestamp = clock();

  // resnet block 1
  reflection_pad_2d(buf.r0_3, buf.r1_0, 1);
  conv2d(buf.r1_0, cycleGAN.m_11.r_1.weight, cycleGAN.m_11.r_1.bias, buf.r1_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r1_1, cycleGAN.m_11.r_2.mean, cycleGAN.m_11.r_2.variance);
  relu(buf.r1_1);
  reflection_pad_2d(buf.r1_1, buf.r1_2, 1);
  conv2d(buf.r1_2, cycleGAN.m_11.r_5.weight, cycleGAN.m_11.r_5.bias, buf.r1_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r1_3, cycleGAN.m_11.r_6.mean, cycleGAN.m_11.r_6.variance);
  add(buf.r0_3, buf.r1_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 1\n", timeSeconds);
  timestamp = clock();

  // resnet block 2
  reflection_pad_2d(buf.r1_3, buf.r2_0, 1);
  conv2d(buf.r2_0, cycleGAN.m_12.r_1.weight, cycleGAN.m_12.r_1.bias, buf.r2_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r2_1, cycleGAN.m_12.r_2.mean, cycleGAN.m_12.r_2.variance);
  relu(buf.r2_1);
  reflection_pad_2d(buf.r2_1, buf.r2_2, 1);
  conv2d(buf.r2_2, cycleGAN.m_12.r_5.weight, cycleGAN.m_12.r_5.bias, buf.r2_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r2_3, cycleGAN.m_12.r_6.mean, cycleGAN.m_12.r_6.variance);
  add(buf.r1_3, buf.r2_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 2\n", timeSeconds);
  timestamp = clock();

  // resnet block 3
  reflection_pad_2d(buf.r2_3, buf.r3_0, 1);
  conv2d(buf.r3_0, cycleGAN.m_13.r_1.weight, cycleGAN.m_13.r_1.bias, buf.r3_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r3_1, cycleGAN.m_13.r_2.mean, cycleGAN.m_13.r_2.variance);
  relu(buf.r3_1);
  reflection_pad_2d(buf.r3_1, buf.r3_2, 1);
  conv2d(buf.r3_2, cycleGAN.m_13.r_5.weight, cycleGAN.m_13.r_5.bias, buf.r3_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r3_3, cycleGAN.m_13.r_6.mean, cycleGAN.m_13.r_6.variance);
  add(buf.r2_3, buf.r3_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 3\n", timeSeconds);
  timestamp = clock();

  // resnet block 4
  reflection_pad_2d(buf.r3_3, buf.r4_0, 1);
  conv2d(buf.r4_0, cycleGAN.m_14.r_1.weight, cycleGAN.m_14.r_1.bias, buf.r4_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r4_1, cycleGAN.m_14.r_2.mean, cycleGAN.m_14.r_2.variance);
  relu(buf.r4_1);
  reflection_pad_2d(buf.r4_1, buf.r4_2, 1);
  conv2d(buf.r4_2, cycleGAN.m_14.r_5.weight, cycleGAN.m_14.r_5.bias, buf.r4_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r4_3, cycleGAN.m_14.r_6.mean, cycleGAN.m_14.r_6.variance);
  add(buf.r3_3, buf.r4_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 4\n", timeSeconds);
  timestamp = clock();

  // resnet block 5
  reflection_pad_2d(buf.r4_3, buf.r5_0, 1);
  conv2d(buf.r5_0, cycleGAN.m_15.r_1.weight, cycleGAN.m_15.r_1.bias, buf.r5_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r5_1, cycleGAN.m_15.r_2.mean, cycleGAN.m_15.r_2.variance);
  relu(buf.r5_1);
  reflection_pad_2d(buf.r5_1, buf.r5_2, 1);
  conv2d(buf.r5_2, cycleGAN.m_15.r_5.weight, cycleGAN.m_15.r_5.bias, buf.r5_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r5_3, cycleGAN.m_15.r_6.mean, cycleGAN.m_15.r_6.variance);
  add(buf.r4_3, buf.r5_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 5\n", timeSeconds);
  timestamp = clock();

  // resnet block 6
  reflection_pad_2d(buf.r5_3, buf.r6_0, 1);
  conv2d(buf.r6_0, cycleGAN.m_16.r_1.weight, cycleGAN.m_16.r_1.bias, buf.r6_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r6_1, cycleGAN.m_16.r_2.mean, cycleGAN.m_16.r_2.variance);
  relu(buf.r6_1);
  reflection_pad_2d(buf.r6_1, buf.r6_2, 1);
  conv2d(buf.r6_2, cycleGAN.m_16.r_5.weight, cycleGAN.m_16.r_5.bias, buf.r6_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r6_3, cycleGAN.m_16.r_6.mean, cycleGAN.m_16.r_6.variance);
  add(buf.r5_3, buf.r6_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 6\n", timeSeconds);
  timestamp = clock();

  // resnet block 7
  reflection_pad_2d(buf.r6_3, buf.r7_0, 1);
  conv2d(buf.r7_0, cycleGAN.m_17.r_1.weight, cycleGAN.m_17.r_1.bias, buf.r7_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r7_1, cycleGAN.m_17.r_2.mean, cycleGAN.m_17.r_2.variance);
  relu(buf.r7_1);
  reflection_pad_2d(buf.r7_1, buf.r7_2, 1);
  conv2d(buf.r7_2, cycleGAN.m_17.r_5.weight, cycleGAN.m_17.r_5.bias, buf.r7_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r7_3, cycleGAN.m_17.r_6.mean, cycleGAN.m_17.r_6.variance);
  add(buf.r6_3, buf.r7_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 7\n", timeSeconds);
  timestamp = clock();

  // resnet block 8
  reflection_pad_2d(buf.r7_3, buf.r8_0, 1);
  conv2d(buf.r8_0, cycleGAN.m_18.r_1.weight, cycleGAN.m_18.r_1.bias, buf.r8_1, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r8_1, cycleGAN.m_18.r_2.mean, cycleGAN.m_18.r_2.variance);
  relu(buf.r8_1);
  reflection_pad_2d(buf.r8_1, buf.r8_2, 1);
  conv2d(buf.r8_2, cycleGAN.m_18.r_5.weight, cycleGAN.m_18.r_5.bias, buf.r8_3, 256, 66, 66, 256, 1, 3, 3);
  instance_norm(buf.r8_3, cycleGAN.m_18.r_6.mean, cycleGAN.m_18.r_6.variance);
  add(buf.r7_3, buf.r8_3);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for resnet 8\n", timeSeconds);
  timestamp = clock();

  // conv transpose
  conv_transpose_2d(buf.r8_3, cycleGAN.m_19.weight, cycleGAN.m_19.bias, buf.o_4, 256, 64, 64, 128, 2, 3, 3, 1, 1);
  instance_norm(buf.o_4, cycleGAN.m_20.mean, cycleGAN.m_20.variance);
  relu(buf.o_4);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for conv transpose\n", timeSeconds);
  timestamp = clock();

  conv_transpose_2d(buf.o_4, cycleGAN.m_22.weight, cycleGAN.m_22.bias, buf.o_5, 128, 128, 128, 64, 2, 3, 3, 1, 1);
  instance_norm(buf.o_5, cycleGAN.m_23.mean, cycleGAN.m_23.variance);
  relu(buf.o_5);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for conv transpose\n", timeSeconds);
  timestamp = clock();

  // conv
  reflection_pad_2d(buf.o_5, buf.o_6, 3);
  conv2d(buf.o_6, cycleGAN.m_26.weight, cycleGAN.m_26.bias, buf.o_7, 64, 262, 262, 3, 1, 7, 7);
  Tanh(buf.o_7);

  timeSeconds = (clock() - timestamp) / (double) CLOCKS_PER_SEC;
  printf("%f s for conv\n", timeSeconds);
  
  timeSeconds = (clock() - begin) / (double) CLOCKS_PER_SEC;
  printf("total %f s\n", timeSeconds);

  return 0;
}