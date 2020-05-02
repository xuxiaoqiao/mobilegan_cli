#ifndef SEQ_LAYERS_HPP
#define SEQ_LAYERS_HPP

#include <vector>
#include "seq/tensor.hpp"

void reflection_pad_2d(const Tensor3D &input, Tensor3D &output, int padding);

void zero_pad_2d(const Tensor3D &input, Tensor3D &output, int padding);

void relu(Tensor3D &data);

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

#endif  // SEQ_LAYERS_HPP