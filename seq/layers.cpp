#include "seq/tensor.hpp"
#include <math.h>

using namespace std;

namespace seq{
void reflection_pad_2d(const Tensor3D &input, Tensor3D &output, int padding) {
  int channel_range = input.shape()[0];
  int input_height = input.shape()[1];
  int input_width = input.shape()[2];
  #ifndef NDEBUG
  int output_height = input.shape().at(1) + padding * 2;
  int output_width = input.shape().at(2) + padding * 2;
  assert(output_height == output.shape().at(1)
             && output_width == output.shape().at(2));
  #endif
  for (int c = 0; c < channel_range; c++) {
    for (int h = padding; h < input_height + padding; h++) {
      for (int w = padding; w < input_width + padding; w++) {
        output(c, h, w) = input(c, h - padding, w - padding);
      }
    }
  }
  // reflect padding
  for (int c = 0; c < channel_range; c++) {
    // top rows
    for (int h = 0; h < padding; h++) {
      for (int w = 0; w < padding; w++) {
        output(c, h, w) = output(c, 2 * padding - h, 2 * padding - w);
      }
      for (int w = padding; w < input_width + padding; w++) {
        output(c, h, w) = output(c, 2 * padding - h, w);
      }
      for (int w = input_width + padding; w < input_width + padding * 2; w++) {
        output(c, h, w) =
            output(c, 2 * padding - h, 2 * (input_width + padding - 1) - w);
      }
    }
    // middle rows
    for (int h = padding; h < input_height + padding; h++) {
      for (int w = 0; w < padding; w++) {
        output(c, h, w) = output(c, h, 2 * padding - w);
      }
      for (int w = input_width + padding; w < input_width + padding * 2; w++) {
        output(c, h, w) = output(c, h, 2 * (input_width + padding - 1) - w);
      }
    }
    // bottom rows
    for (int h = input_height + padding; h < input_height + padding * 2; h++) {
      for (int w = 0; w < padding; w++) {
        output(c, h, w) =
            output(c, 2 * (input_height + padding - 1) - h, 2 * padding - w);
      }
      for (int w = padding; w < input_width + padding; w++) {
        output(c, h, w) = output(c, 2 * (input_height + padding - 1) - h, w);
      }
      for (int w = input_width + padding; w < input_width + padding * 2; w++) {
        output(c, h, w) = output(c,
                                 2 * (input_height + padding - 1) - h,
                                 2 * (input_width + padding - 1) - w);
      }
    }
  }
}

void zero_pad_2d(const Tensor3D &input, Tensor3D &output, int padding) {
  int channel_range = input.shape()[0];
  int input_height = input.shape()[1];
  int input_width = input.shape()[2];
  #ifndef NDEBUG
  int output_height = input.shape().at(1) + padding * 2;
  int output_width = input.shape().at(2) + padding * 2;
  assert(output_height == output.shape().at(1)
             && output_width == output.shape().at(2));
  #endif
  for (int c = 0; c < channel_range; c++) {
    for (int h = padding; h < input_height + padding; h++) {
      for (int w = padding; w < input_width + padding; w++) {
        output(c, h, w) = input(c, h - padding, w - padding);
      }
    }
  }
  // zero padding
  for (int c = 0; c < channel_range; c++) {
    // top rows
    for (int h = 0; h < padding; h++) {
      for (int w = 0; w < input_width + padding * 2; w++) {
        output(c, h, w) = 0;
      }
    }
    // middle rows
    for (int h = padding; h < input_height + padding; h++) {
      for (int w = 0; w < padding; w++) {
        output(c, h, w) = 0;
      }
      for (int w = input_width + padding; w < input_width + padding * 2; w++) {
        output(c, h, w) = 0;
      }
    }
    // bottom rows
    for (int h = input_height + padding; h < input_height + padding * 2; h++) {
      for (int w = 0; w < input_width + padding * 2; w++) {
        output(c, h, w) = 0;
      }
    }
  }
}

void conv2d(const Tensor3D &input,
            const Tensor4D &weight,
            const vector<float> &bias,
            Tensor3D &output,
            int input_channel,
            int input_height,
            int input_width,
            int out_channel,
            int stride,
            int kernel_height,
            int kernel_width) {
  int out_height =
      (input_height - (kernel_height - 1) - 1) / stride + 1;
  int out_width =
      (input_width - (kernel_width - 1) - 1) / stride + 1;
  for (int c = 0; c < out_channel; c++) {
    for (int h = 0; h < out_height; h++) {
      for (int w = 0; w < out_width; w++) {
        float sum = 0;
        for (int ci = 0; ci < input_channel; ci++) {
          // hi, wi --> left top of kernel
          int hi = h * stride;
          int wi = w * stride;
          for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
              sum += weight(c, ci, i, j) * input(ci, hi + i, wi + j);
            }
          }
        }
        sum += bias[c];
        output(c, h, w) = sum;
      }
    }
  }
}

void conv_transpose_2d(const Tensor3D &input,
            const Tensor4D &weight,
            const vector<float> &bias,
            Tensor3D &output,
            int input_channel,
            int input_height,
            int input_width,
            int out_channel,
            int stride,
            int kernel_height,
            int kernel_width,
            int padding,
            int out_padding) {
  int out_height = (input_height-1)*stride - 2*padding + (kernel_height-1) + out_padding + 1;
  int out_width = (input_width-1)*stride - 2*padding + (kernel_width-1) + out_padding + 1;
  int input_pad_h = kernel_height - 1 - padding;
  int input_pad_w = kernel_width - 1 - padding;

  // add appropriate padding to input
  Tensor3D tmp{input_channel, (input_height-1)*stride+1+input_pad_h*2+out_padding, 
               (input_width-1)*stride+1+input_pad_w*2+out_padding};
  for (int c = 0; c < input_channel; c++) {
    for (int h = input_pad_h, hi = 0; h < (input_height-1)*stride+1+input_pad_h; h+=stride, hi++) {
      for (int w = input_pad_w, wi = 0; w < (input_width-1)*stride+1+input_pad_w; w+=stride, wi++) {
        tmp(c, h, w) = input(c, hi, wi);
      }
    }
  }

  // do convolution
  for (int c = 0; c < out_channel; c++) {
    for (int h = 0; h < out_height; h++) {
      for (int w = 0; w < out_width; w++) {
        double sum = 0;
        for (int ci = 0; ci < input_channel; ci++) {
          // hi, wi --> left top of kernel
          int hi = h + input_pad_h - kernel_height/2;
          int wi = w + input_pad_w - kernel_width/2;
          for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
              sum += weight(ci, c, kernel_height-1-i, kernel_width-1-j) * tmp(ci, hi+i, wi+j);
            }
          }
        }
        sum += bias[c];
        output(c, h, w) = sum;
      }
    }
  }
}

void instance_norm(Tensor3D &input, const vector<float> &mean, 
                   const vector<float> &variance) {
  double eps = 1e-5;
  int channel = input.shape().at(0);
  int height = input.shape().at(1);
  int width = input.shape().at(2);
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        input(c, h, w) = (input(c, h, w) - mean[c]) / sqrt(variance[c] + eps);
      }
    }
  }
}

void relu(Tensor3D &data) {
  int channel = data.shape().at(0);
  int height = data.shape().at(1);
  int width = data.shape().at(2);
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        data(c, h, w) = data(c, h, w) >= 0 ? data(c, h, w) : 0;
      }
    }
  }
}

// Tanh starts with capital 'T', distinguishing it with c library call "tanh"
void Tanh(Tensor3D &data) {
  int channel = data.shape().at(0);
  int height = data.shape().at(1);
  int width = data.shape().at(2);
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        data(c, h, w) = tanh(data(c, h, w));
      }
    }
  }
}

void add(Tensor3D &src, Tensor3D &dst) {
  int channel = dst.shape().at(0);
  int height = dst.shape().at(1);
  int width = dst.shape().at(2);
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        dst(c, h, w) += src(c, h, w);
      }
    }
  }
}
}