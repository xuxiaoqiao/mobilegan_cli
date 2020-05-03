#ifndef SEQ_APP_HPP
#define SEQ_APP_HPP

#include "seq/layers.hpp"
#include "seq/model.hpp"

struct gan_buffer_t {
  Tensor3D o_0{3, 262, 262};
  Tensor3D o_1{64, 256, 256};
  Tensor3D o_1_p{64, 258, 258};
  Tensor3D o_2{128, 128, 128};
  Tensor3D o_2_p{128, 130, 130};
  Tensor3D o_3{256, 64, 64};
  Tensor3D r0_0{256, 66, 66};
  Tensor3D r0_1{256, 64, 64};
  Tensor3D r0_2{256, 66, 66};
  Tensor3D r0_3{256, 64, 64};
  Tensor3D r1_0{256, 66, 66};
  Tensor3D r1_1{256, 64, 64};
  Tensor3D r1_2{256, 66, 66};
  Tensor3D r1_3{256, 64, 64};
  Tensor3D r2_0{256, 66, 66};
  Tensor3D r2_1{256, 64, 64};
  Tensor3D r2_2{256, 66, 66};
  Tensor3D r2_3{256, 64, 64};
  Tensor3D r3_0{256, 66, 66};
  Tensor3D r3_1{256, 64, 64};
  Tensor3D r3_2{256, 66, 66};
  Tensor3D r3_3{256, 64, 64};
  Tensor3D r4_0{256, 66, 66};
  Tensor3D r4_1{256, 64, 64};
  Tensor3D r4_2{256, 66, 66};
  Tensor3D r4_3{256, 64, 64};
  Tensor3D r5_0{256, 66, 66};
  Tensor3D r5_1{256, 64, 64};
  Tensor3D r5_2{256, 66, 66};
  Tensor3D r5_3{256, 64, 64};
  Tensor3D r6_0{256, 66, 66};
  Tensor3D r6_1{256, 64, 64};
  Tensor3D r6_2{256, 66, 66};
  Tensor3D r6_3{256, 64, 64};
  Tensor3D r7_0{256, 66, 66};
  Tensor3D r7_1{256, 64, 64};
  Tensor3D r7_2{256, 66, 66};
  Tensor3D r7_3{256, 64, 64};
  Tensor3D r8_0{256, 66, 66};
  Tensor3D r8_1{256, 64, 64};
  Tensor3D r8_2{256, 66, 66};
  Tensor3D r8_3{256, 64, 64};
  Tensor3D o_4{128, 128, 128};
  Tensor3D o_5{64, 256, 256};
  Tensor3D o_6{64, 262, 262};
  Tensor3D o_7{3, 256, 256};
};

int run(const Tensor3D &input, Tensor3D &output, gan_buffer_t &buf, model &cycleGAN);

#endif