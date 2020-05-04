#include "seq/layers.hpp"
#include "seq/unitTest.hpp"
#include <assert.h> 

namespace seq{
void test_reflection_pad_2d() {
    int C = 1;
    int H = 3;
    int W = 3;
    int padding = 2;
    Tensor3D input{C, H, W};
    Tensor3D output{C, H+2*padding, W+2*padding};
    float data[3][3] = {{0,1,2},
                         {3,4,5},
                         {6,7,8}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[h][w];
            }
        }
    }
    reflection_pad_2d(input, output, padding);
    output.print();
    /*
    8 7 6 7 8 7 6 
    5 4 3 4 5 4 3 
    2 1 0 1 2 1 0 
    5 4 3 4 5 4 3 
    8 7 6 7 8 7 6 
    5 4 3 4 5 4 3 
    2 1 0 1 2 1 0 
    */
}

void test_zero_pad_2d() {
    int C = 3;
    int H = 5;
    int W = 5;
    int padding = 1;
    Tensor3D input{C, H, W};
    Tensor3D output{C, H+2*padding, W+2*padding};
    float data[3][3] = {{0,1,2},
                         {3,4,5},
                         {6,7,8}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[h][w];
            }
        }
    }
    zero_pad_2d(input, output, padding);
    output.print();
    /*
    0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 
    0 0 0 1 2 0 0 
    0 0 3 4 5 0 0 
    0 0 6 7 8 0 0 
    0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 
    */
}

void test_conv_2d() {
    int C = 3;
    int H = 5;
    int W = 5;
    int padding = 1;
    Tensor3D input{C, H, W};
    Tensor3D input_pad{C, H+2*padding, W+2*padding};
    float data[3][5][5] = {{{0,1,1,2,2},
                             {1,2,1,2,0},
                             {1,2,2,2,0},
                             {0,0,1,2,2},
                             {0,2,2,0,1}}, 
                            {{0,1,0,1,1},
                             {2,2,2,2,2},
                             {2,1,1,0,0},
                             {1,2,2,1,1},
                             {0,2,0,0,0}}, 
                            {{0,1,1,2,0},
                             {0,2,0,1,0},
                             {2,0,1,1,2},
                             {1,1,0,2,2},
                             {1,2,1,0,2}}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[c][h][w];
            }
        }
    }
    zero_pad_2d(input, input_pad, padding);

    int C_out = 2;
    int kernel_H = 3;
    int kernel_W = 3;
    int stride = 2;
    int H_out = (H + padding*2 - (kernel_H - 1) - 1) / stride + 1;
    int W_out = (W + padding*2 - (kernel_W - 1) - 1) / stride + 1;
    Tensor4D kernel{C_out, C, kernel_H, kernel_W};
    std::vector<float> bias{1, 0};
    Tensor3D output{C_out, H_out, W_out};

    float k_data[2][3][3][3] = {{{{0,-1,0},
                                   {1,-1,1},
                                   {0,1,0}},
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
                                  {{-1,0,0},
                                   {-1,0,-1},
                                   {0,0,-1}}}};
    for (int co = 0; co < C_out; co++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < kernel_H; h++) {
                for (int w = 0; w < kernel_W; w++) {
                    kernel(co, c, h, w) = k_data[co][c][h][w];
                }
            }
        }
    }
    conv2d(input_pad, kernel, bias, output, C, H+2*padding, W+2*padding, C_out, 
           stride, kernel_H, kernel_W);
    output.print();
    /*
    [2 4 -1 
     7 6 9 
     9 -3 4] 
    [-3 -6 -4 
     -10 -11 -8 
     -3 -4 -5] 
    */
}

void test_conv_transpose_2d() {
    int C = 2;
    int H = 2;
    int W = 2;
    Tensor3D input{C, H, W};
    double data[2][2][2] = {{{3,3},
                             {1,1}},
                            {{1,2},
                             {3,4}}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[c][h][w];
            }
        }
    }

    int C_out = 2;
    int kernel_H = 3;
    int kernel_W = 3;
    int stride = 2;
    int padding = 1;
    int out_padding = 1;
    int H_out = (H-1)*stride - 2*padding + (kernel_H-1) + out_padding + 1;
    int W_out = (W-1)*stride - 2*padding + (kernel_W-1) + out_padding + 1;
    Tensor4D kernel{C, C_out, kernel_H, kernel_W};
    std::vector<float> bias{1, 2};
    Tensor3D output{C_out, H_out, W_out};
    double k_data[2][2][3][3] = {{{{1,2,3},
                                   {0,1,0},
                                   {2,1,2}},
                                  {{1,2,3},
                                   {4,5,6},
                                   {7,8,9}}},
                                 
                                 {{{1,1,1},
                                   {1,1,1},
                                   {1,1,1}},
                                  {{0,0,0},
                                   {0,0,0},
                                   {0,0,0}}}};
    for (int c = 0; c < C; c++) {
        for (int co = 0; co < C_out; co++) {
            for (int h = 0; h < kernel_H; h++) {
                for (int w = 0; w < kernel_W; w++) {
                    kernel(c, co, h, w) = k_data[c][co][h][w];
                }
            }
        }
    }
    conv_transpose_2d(input, kernel, bias, output, C, H, W, C_out,
                      stride, kernel_H, kernel_W, padding, out_padding);
    output.print();
    /*
    [5 4 6 3 
    10 27 12 16 
    5 8 6 5 
    5 12 6 7] 
    [17 32 17 20 
    28 54 28 32 
    7 12 7 8 
    10 18 10 11] 
    */
}

void test_instance_norm() {
    int C = 3;
    int H = 2;
    int W = 1;
    Tensor3D input{C, H, W};
    double data[3][2][1] = {{{1},{2}},{{3},{4}},{{5},{6}}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[c][h][w];
            }
        }
    }
    std::vector<float> mean{0, 1, 2};
    std::vector<float> variance{0.5, 0.3, 1.2};
    instance_norm(input, mean, variance);
    input.print();
    /*
    1.4142 
    2.8284 
    3.65142 
    5.47713 
    2.7386 
    3.65147 
    */
}

void test_relu() {
    int C = 3;
    int H = 2;
    int W = 1;
    Tensor3D input{C, H, W};
    float data[3][2][1] = {{{1},{-2}},{{0.0003},{-0.0004}},{{5},{6}}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[c][h][w];
            }
        }
    }
    relu(input);
    input.print();
    /*
    1 
    0 
    0.0003 
    0 
    5 
    6 
    */
}

void test_Tanh() {
    int C = 3;
    int H = 2;
    int W = 1;
    Tensor3D input{C, H, W};
    double data[3][2][1] = {{{1},{2}},{{3},{4}},{{5},{6}}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[c][h][w];
            }
        }
    }
    Tanh(input);
    input.print();
    /*
    0.761594 
    0.964028 
    0.995055 
    0.999329 
    0.999909 
    0.999988 
    */
}

void test_conv_2d_2() {
    int C = 4;
    int H = 6;
    int W = 6;
    Tensor3D input{C, H, W};
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
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[c][h][w];
            }
        }
    }

    int C_out = 4;
    int kernel_H = 3;
    int kernel_W = 3;
    int stride = 1;
    int H_out = (H - (kernel_H - 1) - 1) / stride + 1;
    int W_out = (W - (kernel_W - 1) - 1) / stride + 1;
    Tensor4D kernel{C_out, C, kernel_H, kernel_W};
    std::vector<float> bias{1, 0, 1, 0};
    Tensor3D output{C_out, H_out, W_out};

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
    for (int co = 0; co < C_out; co++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < kernel_H; h++) {
                for (int w = 0; w < kernel_W; w++) {
                    kernel(co, c, h, w) = k_data[co][c][h][w];
                }
            }
        }
    }
    conv2d(input, kernel, bias, output, C, H, W, C_out, 
           stride, kernel_H, kernel_W);
    output.print();
}

void test_conv_transpose_2d_2() {
    int C = 4;
    int H = 2;
    int W = 2;
    Tensor3D input{C, H, W};
    double data[4][2][2] = {{{3., 3.},
                            {1., 1.}},
                            {{3., 3.},
                            {1., 1.}},
                            {{1., 2.},
                            {3., 4.}},
                            {{1., 2.},
                            {3., 4.}}};
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                input(c, h, w) = data[c][h][w];
            }
        }
    }

    int C_out = 4;
    int kernel_H = 3;
    int kernel_W = 3;
    int stride = 2;
    int padding = 1;
    int out_padding = 1;
    int H_out = (H-1)*stride - 2*padding + (kernel_H-1) + out_padding + 1;
    int W_out = (W-1)*stride - 2*padding + (kernel_W-1) + out_padding + 1;
    Tensor4D kernel{C, C_out, kernel_H, kernel_W};
    std::vector<float> bias{1, 0, 1, 0};
    Tensor3D output{C_out, H_out, W_out};
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
    for (int c = 0; c < C; c++) {
        for (int co = 0; co < C_out; co++) {
            for (int h = 0; h < kernel_H; h++) {
                for (int w = 0; w < kernel_W; w++) {
                    kernel(c, co, h, w) = k_data[c][co][h][w];
                }
            }
        }
    }
    conv_transpose_2d(input, kernel, bias, output, C, H, W, C_out,
                      stride, kernel_H, kernel_W, padding, out_padding);
    output.print();
    /*
    [5 4 6 3 
    10 27 12 16 
    5 8 6 5 
    5 12 6 7] 
    [17 32 17 20 
    28 54 28 32 
    7 12 7 8 
    10 18 10 11] 
    */
}

}