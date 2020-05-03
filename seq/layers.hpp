#ifndef SEQ_LAYERS_HPP
#define SEQ_LAYERS_HPP

#include <vector>
#include "seq/tensor.hpp"

void reflection_pad_2d(const Tensor3D &input, Tensor3D &output, int padding);

void zero_pad_2d(const Tensor3D &input, Tensor3D &output, int padding);

void relu(Tensor3D &data);

void Tanh(Tensor3D &data);

void add(Tensor3D &src, Tensor3D &dst);

// note: padding is not supported (add a zero_pad_2d/reflection_pad_2d)
void conv2d(const Tensor3D &input,
            const Tensor4D &weight,
            const std::vector<float> &bias,
            Tensor3D &output,
            int input_channel,
            int input_height,
            int input_width,
            int out_channel,
            int stride,
            int kernel_height,
            int kernel_width);

void conv_transpose_2d(const Tensor3D &input,
            const Tensor4D &weight,
            const std::vector<float> &bias,
            Tensor3D &output,
            int input_channel,
            int input_height,
            int input_width,
            int out_channel,
            int stride,
            int kernel_height,
            int kernel_width,
            int padding,
            int out_padding);

void instance_norm(Tensor3D &input, const std::vector<float> &mean, 
                   const std::vector<float> &variance);

#endif  // SEQ_LAYERS_HPP