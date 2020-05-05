#include <iostream>
#include <vector>
#include <CL/cl.h>
#include "parallel/unittest.hpp"
#include "parallel/runtime.hpp"
#include "parallel/layers.hpp"

using namespace std;

void trial_run_conv2d(cl_context context, cl_command_queue queue) {
  constexpr int INPUT_CHANNEL_BLK = 64;
  constexpr int INPUT_HW = 66;
  constexpr int OUTPUT_CHANNEL_BLK = 64;
  constexpr int OUTPUT_HW = 64;
  constexpr int STRIDE = 1;

  vector<float> input(INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4);
  vector<float> weight(OUTPUT_CHANNEL_BLK * INPUT_CHANNEL_BLK * 3 * 3 * 4 * 4); // C_out, C_in, Kh, Kw, 4c_out, 4c_in
  vector<float> output(OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4);

  for (int c_blk = 0; c_blk < INPUT_CHANNEL_BLK; c_blk++) {
    for (int h = 0; h < INPUT_HW; h++) {
      for (int w = 0; w < INPUT_HW; w++) {
        for (int c_itr = 0; c_itr < 4; c_itr++) {
          int idx = c_blk * INPUT_HW * INPUT_HW * 4 + h * INPUT_HW * 4 + w * 4 + c_itr;
          input[idx] = (float) (h*INPUT_HW+w);
        }
      }
    }
  }

  for (int oc_blk = 0; oc_blk < OUTPUT_CHANNEL_BLK; oc_blk++) {
    for (int ic_blk = 0; ic_blk < INPUT_CHANNEL_BLK; ic_blk++) {
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          for (int i = 0; i < 16; i++) {
            int idx = oc_blk * INPUT_CHANNEL_BLK * 3 * 3 * 16 +
                ic_blk * 3 * 3 * 16 +
                kh * 3 * 16 +
                kw * 16 + i;
            if (kh == 1 && kw == 1) {
              weight[idx] = 1.0/(INPUT_CHANNEL_BLK*4);
            } else {
              weight[idx] = 0;
            }
          }
        }
      }
    }
  }

  cl_mem input_clbuf = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4,
                                      input.data(),
                                      NULL);
  cl_mem weight_clbuf = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * INPUT_CHANNEL_BLK * 3 * 3 * 4 * 4,
                                       weight.data(),
                                       NULL);
  cl_mem output_clbuf = clCreateBuffer(context,
                                       CL_MEM_WRITE_ONLY,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4,
                                       NULL,
                                       NULL);
  conv2d_exec_async(queue,
                    input_clbuf,
                    weight_clbuf,
                    nullptr,
                    output_clbuf,
                    nullptr,
                    nullptr,
                    INPUT_CHANNEL_BLK * 4,
                    INPUT_HW,
                    INPUT_HW,
                    OUTPUT_CHANNEL_BLK * 4,
                    OUTPUT_HW,
                    OUTPUT_HW,
                    STRIDE,
                    3,
                    3,
                    false,
                    activation::NONE);

  cl_int errNum;
  errNum = clEnqueueReadBuffer(queue, output_clbuf, CL_TRUE,
                               0, OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4 * sizeof(float), output.data(),
                               0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error reading result buffer." << std::endl;
    std::abort();
  }

  // Output the result buffer
  for (int c_blk = 0; c_blk < OUTPUT_CHANNEL_BLK; c_blk++) {
    for (int h = 0; h < OUTPUT_HW; h++) {
      for (int w = 0; w < OUTPUT_HW; w++) {
        printf("[c_blk: %d, h: %d, w: %d]: ", c_blk, h, w);
        for (int c_itr = 0; c_itr < 4; c_itr++) {
          int idx = c_blk * OUTPUT_HW * OUTPUT_HW * 4 + h * OUTPUT_HW * 4 + w * 4 + c_itr;
          std::cout << output[idx] << " ";
        }
        printf("\n");
      }
    }
  }
  std::cout << std::endl;
  std::cout << "Executed program succesfully." << std::endl;
  clReleaseMemObject(input_clbuf);
  clReleaseMemObject(weight_clbuf);
  clReleaseMemObject(output_clbuf);
}

void test_conv2d(cl_context context, cl_command_queue queue) {
  constexpr int INPUT_CHANNEL_BLK = 1;
  constexpr int INPUT_HW = 6;
  constexpr int OUTPUT_CHANNEL_BLK = 1;
  constexpr int OUTPUT_HW = 4;
  constexpr int STRIDE = 1;

  vector<float> input(INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4);
  vector<float> weight(OUTPUT_CHANNEL_BLK * INPUT_CHANNEL_BLK * 3 * 3 * 4 * 4); // C_out, C_in, Kh, Kw, 4c_in, 4c_out
  vector<float> bias{1, 0, 1, 0};
  vector<float> output(OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4);

  float data[4][6][6] = {{{0,1,1,2,2,0},
                          {1,2,1,2,0,0},
                          {1,2,2,2,0,0},
                          {0,0,1,2,2,0},
                          {0,0,1,2,2,0},
                          {0,2,2,0,1,0}}, 
                          {{0,1,0,1,1,0},
                          {2,2,2,2,2,0},
                          {2,1,1,0,0,0},
                          {1,2,2,1,1,0},
                          {1,2,2,1,1,0},
                          {0,2,0,0,0,0}},
                          {{1,1,1,1,1,0},
                          {1,1,1,1,1,0},
                          {1,1,1,1,1,0},
                          {1,1,1,1,1,0},
                          {1,1,1,1,1,0},
                          {1,1,1,1,1,0}}, 
                          {{0,1,1,2,0,0},
                          {0,2,0,1,0,0},
                          {2,0,1,1,2,0},
                          {1,1,0,2,2,0},
                          {1,1,0,2,2,0},
                          {1,2,1,0,2,0}}};

  for (int c_blk = 0; c_blk < INPUT_CHANNEL_BLK; c_blk++) {
    for (int h = 0; h < INPUT_HW; h++) {
      for (int w = 0; w < INPUT_HW; w++) {
        for (int c_itr = 0; c_itr < 4; c_itr++) {
          int idx = c_blk * INPUT_HW * INPUT_HW * 4 + h * INPUT_HW * 4 + w * 4 + c_itr;
          input[idx] = data[c_blk*4+c_itr][h][w];
        }
      }
    }
  }

  float k_data[4][4][3][3] = {{{{0,-1,0},
                                {1,-1,1},
                                {0,1,0}},
                                {{-1,1,1},
                                {-1,1,0},
                                {0,0,0}},
                                {{-1,1,1},
                                {-1,1,0},
                                {0,0,0}},
                                {{1,1,-1},
                                {-1,1,1},
                                {0,1,-1}}},

                                {{{0,-1,0},
                                {1,-1,1},
                                {0,1,0}},
                                {{-1,1,1},
                                {-1,1,0},
                                {0,0,0}},
                                {{-1,1,1},
                                {-1,1,0},
                                {0,0,0}},
                                {{1,1,-1},
                                {-1,1,1},
                                {0,1,-1}}},

                                {{{-1,-1,-1},
                                {-1,1,-1},
                                {-1,0,1}},
                                {{1,-1,0},
                                {1,-1,1},
                                {0,0,-1}},
                                {{-1,1,1},
                                {-1,1,0},
                                {0,0,0}},
                                {{-1,0,0},
                                {-1,0,-1},
                                {0,0,-1}}},

                                {{{-1,-1,-1},
                                {-1,1,-1},
                                {-1,0,1}},
                                {{1,-1,0},
                                {1,-1,1},
                                {0,0,-1}},
                                {{-1,1,1},
                                {-1,1,0},
                                {0,0,0}},
                                {{-1,0,0},
                                {-1,0,-1},
                                {0,0,-1}}}};

  for (int oc_blk = 0; oc_blk < OUTPUT_CHANNEL_BLK; oc_blk++) {
    for (int ic_blk = 0; ic_blk < INPUT_CHANNEL_BLK; ic_blk++) {
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          for (int i = 0; i < 16; i++) {
            int idx = oc_blk * INPUT_CHANNEL_BLK * 3 * 3 * 16 +
                ic_blk * 3 * 3 * 16 +
                kh * 3 * 16 +
                kw * 16 + i;
            weight.at(idx)= k_data[oc_blk*4+(i%4)][ic_blk*4+(i/4)][kh][kw];
          }
        }
      }
    }
  }

  cl_mem input_clbuf = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4,
                                      input.data(),
                                      NULL);
  cl_mem weight_clbuf = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * INPUT_CHANNEL_BLK * 3 * 3 * 4 * 4,
                                       weight.data(),
                                       NULL);
  cl_mem bias_clbuf = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * 4,
                                       bias.data(),
                                       NULL);
  cl_mem output_clbuf = clCreateBuffer(context,
                                       CL_MEM_WRITE_ONLY,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4,
                                       NULL,
                                       NULL);

  conv2d_exec_async(queue,
                    input_clbuf,
                    weight_clbuf,
                    bias_clbuf,
                    output_clbuf,
                    nullptr,
                    nullptr,
                    INPUT_CHANNEL_BLK * 4,
                    INPUT_HW,
                    INPUT_HW,
                    OUTPUT_CHANNEL_BLK * 4,
                    OUTPUT_HW,
                    OUTPUT_HW,
                    STRIDE,
                    3,
                    3,
                    false,
                    activation::NONE);

  cl_int errNum;
  errNum = clEnqueueReadBuffer(queue, output_clbuf, CL_TRUE,
                               0, OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4 * sizeof(float), output.data(),
                               0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error reading result buffer." << std::endl;
    std::abort();
  }

  // Output the result buffer
  for (int c_blk = 0; c_blk < OUTPUT_CHANNEL_BLK; c_blk++) {
    for (int h = 0; h < OUTPUT_HW; h++) {
      for (int w = 0; w < OUTPUT_HW; w++) {
        printf("[c_blk: %d, h: %d, w: %d]: ", c_blk, h, w);
        for (int c_itr = 0; c_itr < 4; c_itr++) {
          int idx = c_blk * OUTPUT_HW * OUTPUT_HW * 4 + h * OUTPUT_HW * 4 + w * 4 + c_itr;
          std::cout << output[idx] << " ";
        }
        printf("\n");
      }
    }
  }
  std::cout << std::endl;
  std::cout << "Executed program succesfully." << std::endl;
  clReleaseMemObject(input_clbuf);
  clReleaseMemObject(weight_clbuf);
  clReleaseMemObject(bias_clbuf);
  clReleaseMemObject(output_clbuf);
  /*
  [[[[  5,   5,   6,   4],
    [  4,   7,   6,   9],
    [  4,   1,   5,   8],
    [ 12,   6,   2,   6]],

    [[  4,   4,   5,   3],
    [  3,   6,   5,   8],
    [  3,   0,   4,   7],
    [ 11,   5,   1,   5]],

    [[  0,  -7,  -6, -10],
    [ -5,  -9,  -5,  -7],
    [ -6,  -8,  -5,  -6],
    [ -1,  -7,  -6,  -7]],

    [[ -1,  -8,  -7, -11],
    [ -6, -10,  -6,  -8],
    [ -7,  -9,  -6,  -7],
    [ -2,  -8,  -7,  -8]]]]
  */
}

void test_convtranspose_2d(cl_context context, cl_command_queue queue) {
  constexpr int INPUT_CHANNEL_BLK = 1;
  constexpr int INPUT_HW = 2;
  constexpr int OUTPUT_CHANNEL_BLK =  1;
  constexpr int OUTPUT_HW =  4;

  vector<float> input(INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4);
  vector<float> weight(INPUT_CHANNEL_BLK * OUTPUT_CHANNEL_BLK * 3 * 3 * 4 * 4); // C_in, C_out, Kh, Kw, 4c_in, 4c_out
  vector<float> bias{1, 0, 1, 0};
  vector<float> output(OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4);

  float data[4][2][2] = {{{3., 3.},
                          {1., 1.}},
                          {{3., 3.},
                          {1., 1.}},
                          {{1., 2.},
                          {3., 4.}},
                          {{1., 2.},
                          {3., 4.}}};

  for (int c_blk = 0; c_blk < INPUT_CHANNEL_BLK; c_blk++) {
    for (int h = 0; h < INPUT_HW; h++) {
      for (int w = 0; w < INPUT_HW; w++) {
        for (int c_itr = 0; c_itr < 4; c_itr++) {
          int idx = c_blk * INPUT_HW * INPUT_HW * 4 + h * INPUT_HW * 4 + w * 4 + c_itr;
          input[idx] = data[c_blk*4+c_itr][h][w];
        }
      }
    }
  }

  float k_data[4][4][3][3] = {{{{1., 2., 3.},
                                {0., 1., 0.},
                                {2., 1., 2.}},
                                {{1., 2., 3.},
                                {4., 5., 6.},
                                {7., 8., 9.}},
                                {{1., 2., 3.},
                                {0., 1., 0.},
                                {2., 1., 2.}},
                                {{1., 2., 3.},
                                {4., 5., 6.},
                                {7., 8., 9.}}},
                                {{{1., 2., 3.},
                                {0., 1., 0.},
                                {2., 1., 2.}},
                                {{1., 2., 3.},
                                {4., 5., 6.},
                                {7., 8., 9.}},
                                {{1., 2., 3.},
                                {0., 1., 0.},
                                {2., 1., 2.}},
                                {{1., 2., 3.},
                                {4., 5., 6.},
                                {7., 8., 9.}}},
                                {{{1., 1., 1.},
                                {1., 1., 1.},
                                {1., 1., 1.}},
                                {{0., 0., 0.},
                                {0., 0., 0.},
                                {0., 0., 0.}},
                                {{1., 1., 1.},
                                {1., 1., 1.},
                                {1., 1., 1.}},
                                {{0., 0., 0.},
                                {0., 0., 0.},
                                {0., 0., 0.}}},
                                {{{1., 1., 1.},
                                {1., 1., 1.},
                                {1., 1., 1.}},
                                {{0., 0., 0.},
                                {0., 0., 0.},
                                {0., 0., 0.}},
                                {{1., 1., 1.},
                                {1., 1., 1.},
                                {1., 1., 1.}},
                                {{0., 0., 0.},
                                {0., 0., 0.},
                                {0., 0., 0.}}}};

  for (int ic_blk = 0; ic_blk < INPUT_CHANNEL_BLK; ic_blk++) {
    for (int oc_blk = 0; oc_blk < OUTPUT_CHANNEL_BLK; oc_blk++) {
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          for (int i = 0; i < 16; i++) {
            int idx = ic_blk * OUTPUT_CHANNEL_BLK * 3 * 3 * 16 +
                oc_blk * 3 * 3 * 16 +
                kh * 3 * 16 +
                kw * 16 + i;
            weight[idx] = k_data[ic_blk*4+(i%4)][oc_blk*4+(i/4)][kh][kw];
          }
        }
      }
    }
  }

  cl_mem input_clbuf = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4,
                                      input.data(),
                                      NULL);
  cl_mem weight_clbuf = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * INPUT_CHANNEL_BLK * 3 * 3 * 4 * 4,
                                       weight.data(),
                                       NULL);
  cl_mem bias_clbuf = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * 4,
                                       bias.data(),
                                       NULL);
  cl_mem output_clbuf = clCreateBuffer(context,
                                       CL_MEM_WRITE_ONLY,
                                       sizeof(float) * OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4,
                                       NULL,
                                       NULL);

  conv2d_transpose_3x3_stride2_norm_relu_exec_async(
                    queue,
                    input_clbuf,
                    weight_clbuf,
                    bias_clbuf,
                    output_clbuf,
                    nullptr,
                    nullptr,
                    INPUT_CHANNEL_BLK * 4,
                    INPUT_HW,
                    INPUT_HW,
                    OUTPUT_CHANNEL_BLK * 4,
                    OUTPUT_HW,
                    OUTPUT_HW,
                    false,
                    activation::NONE);

  cl_int errNum;
  errNum = clEnqueueReadBuffer(queue, output_clbuf, CL_TRUE,
                               0, OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4 * sizeof(float), output.data(),
                               0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Error reading result buffer. errNum: " << errNum << std::endl;
    std::abort();
  }

  // Output the result buffer
  for (int c_blk = 0; c_blk < OUTPUT_CHANNEL_BLK; c_blk++) {
    for (int h = 0; h < OUTPUT_HW; h++) {
      for (int w = 0; w < OUTPUT_HW; w++) {
        printf("[c_blk: %d, h: %d, w: %d]: ", c_blk, h, w);
        for (int c_itr = 0; c_itr < 4; c_itr++) {
          int idx = c_blk * OUTPUT_HW * OUTPUT_HW * 4 + h * OUTPUT_HW * 4 + w * 4 + c_itr;
          std::cout << output[idx] << " ";
        }
        printf("\n");
      }
    }
  }
  std::cout << std::endl;
  std::cout << "Executed program succesfully." << std::endl;
  clReleaseMemObject(input_clbuf);
  clReleaseMemObject(weight_clbuf);
  clReleaseMemObject(bias_clbuf);
  clReleaseMemObject(output_clbuf);
  /*
  [[[[  9.,   7.,  11.,   5.],
    [ 19.,  53.,  23.,  31.],
    [  9.,  15.,  11.,   9.],
    [  9.,  23.,  11.,  13.]],

    [[ 30.,  60.,  30.,  36.],
    [ 52., 104.,  52.,  60.],
    [ 10.,  20.,  10.,  12.],
    [ 16.,  32.,  16.,  18.]],

    [[  9.,   7.,  11.,   5.],
    [ 19.,  53.,  23.,  31.],
    [  9.,  15.,  11.,   9.],
    [  9.,  23.,  11.,  13.]],

    [[ 30.,  60.,  30.,  36.],
    [ 52., 104.,  52.,  60.],
    [ 10.,  20.,  10.,  12.],
    [ 16.,  32.,  16.,  18.]]]]
  */
}

int test_run() {
  cl_context context = nullptr;
  cl_command_queue commandQueue = nullptr;
  cl_device_id device = nullptr;
  cl_int errNum;
  context = CreateContext();
  if (context == nullptr) {
    std::cerr << "Failed to create OpenCL context." << std::endl;
    std::abort();
  }
  commandQueue = CreateCommandQueue(context, &device);
  if (commandQueue == nullptr) {
    std::cerr << "Failed to create OpenCL command queue." << std::endl;
    std::abort();
  }

  init_kernels(context, device);
  // trial_run_conv2d(context, commandQueue);
  // test_conv2d(context, commandQueue);
  test_convtranspose_2d(context, commandQueue);

  return 0;
}
