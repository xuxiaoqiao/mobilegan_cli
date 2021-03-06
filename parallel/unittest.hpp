#ifndef PARALLEL_UNITTEST_HPP
#define PARALLEL_UNITTEST_HPP

#include <CL/cl.h>

void trial_run_conv2d(cl_context context, cl_command_queue queue);
void test_convtranspose_2d(cl_context context, cl_command_queue queue);
int test_run();

#endif // PARALLEL_UNITTEST_HPP