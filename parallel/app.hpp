#ifndef PARALLEL_APP_HPP
#define PARALLEL_APP_HPP


#include "parallel/model.hpp"
namespace parallel {

class gan_buffer_t final {
 public:
  cl_mem o_0;
  cl_mem o_1;
  cl_mem o_1_p;
  cl_mem o_2;
  cl_mem o_2_p;
  cl_mem o_3;
  cl_mem r0_0;
  cl_mem r0_1;
  cl_mem r0_2;
  cl_mem r0_3;
  cl_mem r1_0;
  cl_mem r1_1;
  cl_mem r1_2;
  cl_mem r1_3;
  cl_mem r2_0;
  cl_mem r2_1;
  cl_mem r2_2;
  cl_mem r2_3;
  cl_mem r3_0;
  cl_mem r3_1;
  cl_mem r3_2;
  cl_mem r3_3;
  cl_mem r4_0;
  cl_mem r4_1;
  cl_mem r4_2;
  cl_mem r4_3;
  cl_mem r5_0;
  cl_mem r5_1;
  cl_mem r5_2;
  cl_mem r5_3;
  cl_mem r6_0;
  cl_mem r6_1;
  cl_mem r6_2;
  cl_mem r6_3;
  cl_mem r7_0;
  cl_mem r7_1;
  cl_mem r7_2;
  cl_mem r7_3;
  cl_mem r8_0;
  cl_mem r8_1;
  cl_mem r8_2;
  cl_mem r8_3;
  cl_mem o_4;
  cl_mem o_5;
  cl_mem o_6;
  gan_buffer_t() = delete;
  explicit gan_buffer_t(cl_context context);
  ~gan_buffer_t();
};

int run(cl_mem input, cl_mem output, gan_buffer_t &buf, model &cycleGAN,
        cl_command_queue queue);

// struct profile_data {
//   struct resnet_block_profile {
//     double pad0 = 0.0;
//     double conv0 = 0.0;
//     double pad1 = 0.0;
//     double conv1 = 0.0;
//     double add = 0.0;
//   };
//   int recorded_run = 0;
//   int current_iter = 0;
//   double pad0 = 0.0;
//   double conv0 = 0.0;
//   double pad1 = 0.0;
//   double conv1 = 0.0;
//   double pad2 = 0.0;
//   double conv2 = 0.0;
// };

int example_main();

}


#endif