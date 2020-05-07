#include <iostream>
#include "seq/tensor.hpp"
#include "seq/unitTest.hpp"
#include "seq/model.hpp"
#include "seq/app.hpp"
using namespace std;
using namespace seq;

int main(int argc, char **argv) {
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
  return 0;
}