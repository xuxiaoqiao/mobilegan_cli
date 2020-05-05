#ifndef PARALLEL_MODEL_HPP
#define PARALLEL_MODEL_HPP
#include <vector>
#include <CL/cl.h>
namespace parallel {

/*
    kernel in conv2d is [C_out, C_in, H, W]
    kernel in convtranspose2d is [C_in, C_out, H, W]
*/
class kernel
{
public:
    int C_out;
    int C_in;
    int H;
    int W;
    std::vector<float> weight;
    std::vector<float> bias;
    cl_mem weight_buf;
    cl_mem bias_buf;

    kernel(int c_out, int c_in, int h, int w) 
    : C_out(c_out), C_in(c_in), H(h), W(w) {
        int c_out_tmp = c_out == 3 ? 4 : c_out;
        int c_in_tmp = c_in == 3 ? 4 : c_in;
        weight.resize(c_out_tmp * c_in_tmp * h * w, 0.0);
        bias.resize(c_out_tmp, 0.0);
    }
};

class norm_stat
{
public:
    int C;
    std::vector<float> mean;
    std::vector<float> variance;
    cl_mem mean_buf;
    cl_mem variance_buf;

    norm_stat(int c) 
    : C(c), mean(c, 0.0), variance(c, 0.0) {}
};

class res_block
{
public:
    kernel r_1;
    norm_stat r_2;
    kernel r_5;
    norm_stat r_6;

    res_block(int c, int h, int w) 
    : r_1(c, c, h, w), r_2(c), r_5(c, c, h, w), r_6(c) {}
};

class model
{
public:
    kernel m_1;
    norm_stat m_2;
    kernel m_4;
    norm_stat m_5;
    kernel m_7;
    norm_stat m_8;
    res_block m_10;
    res_block m_11;
    res_block m_12;
    res_block m_13;
    res_block m_14;
    res_block m_15;
    res_block m_16;
    res_block m_17;
    res_block m_18;
    kernel m_19;
    norm_stat m_20;
    kernel m_22;
    norm_stat m_23;
    kernel m_26;

    model() 
    : m_1(64, 3, 7, 7), m_2(64),
      m_4(128, 64, 3, 3), m_5(128),
      m_7(256, 128, 3, 3), m_8(256),
      m_10(256, 3, 3), m_11(256, 3, 3),
      m_12(256, 3, 3), m_13(256, 3, 3),
      m_14(256, 3, 3), m_15(256, 3, 3),
      m_16(256, 3, 3), m_17(256, 3, 3),
      m_18(256, 3, 3),
      m_19(128, 256, 3, 3), m_20(128),
      m_22(64, 128, 3, 3), m_23(64),
      m_26(3, 64, 7, 7) {}

};

void load_model(model &cycleGAN, cl_context context);
void release_model(model &cycleGAN);

}
#endif //PARALLEL_MODEL_HPP
