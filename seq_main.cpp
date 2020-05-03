#include <iostream>
#include "seq/tensor.hpp"
#include "seq/unitTest.hpp"
#include "seq/model.hpp"
#include "seq/app.hpp"
using namespace std;

int main(int argc, char **argv) {
  // test_reflection_pad_2d();
  // test_zero_pad_2d();
  // test_conv_2d();
  // test_conv_transpose_2d();
  // test_instance_norm();
  // test_relu();
  // test_Tanh();
  model cycleGAN;
  load_model(cycleGAN);
  gan_buffer_t buf;
  int C = 3;
  int H = 256;
  int W = 256;
  Tensor3D input(3, 256, 256);
  Tensor3D output(3, 256, 256);
  for (int c = 0; c < C; c++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        input(c, h, w) = (float)rand()/RAND_MAX; // [0.0, 1.0]
      }
    }
  }
  run(input, output, buf, cycleGAN);
//  cycleGAN.m_26.weight.print();
//  for (int i = 0; i < cycleGAN.m_4.C_out; i++) {
//    printf("%f\n", cycleGAN.m_4.bias[i]);
//  }
  return 0;
}