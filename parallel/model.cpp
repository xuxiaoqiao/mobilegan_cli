#include "parallel/model.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
namespace parallel {
void load_1d_array(const char* filepath, std::vector<float> &data) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int count = 0;

    fp = fopen(filepath, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1) {
        char *token, *str, *tofree, *ptr;
        tofree = str = strdup(line);
        data.at(count) = strtof(str, &ptr);
        free(tofree);
        count++;
    }

    fclose(fp);
    if (line)
        free(line);
}

void load_4d_array(const char* filepath, int O, int I, int H, int W, std::vector<float> &data) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int line_count = 0;
    int I_ = (I+1) / 4;

    fp = fopen(filepath, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1) {
        int o = line_count / (I * H);
        int i = (line_count % (I * H)) / H;
        int h = line_count % (I * H) % H;
        int w = 0;
        char *token, *str, *tofree;
        tofree = str = strdup(line);
        while ((token = strsep(&str, " "))) {
            char *ptr;
            // data layout: Cout_blk, Cin_blk, kh, kw, cin4, cout4
            int out_channel_blk_idx = o / 4;
            int in_channel_blk_idx = i / 4;
            int out_channel_quad_idx = o % 4;
            int in_channel_quad_idx = i % 4;
            int idx = out_channel_blk_idx * (I_ * H * W * 16) + 
                      in_channel_blk_idx * (H * W * 16) +
                      h * (W * 16) + w * 16 + in_channel_quad_idx * 4 + 
                      out_channel_quad_idx;
            data.at(idx) = strtof(token, &ptr);
            w++;
        }
        free(tofree);
        line_count++;
    }
    fclose(fp);
    if (line)
        free(line);
}

void load_kernel(char* wfile, char* bfile, kernel &k, cl_context context) {
    load_4d_array(wfile, k.C_out, k.C_in, k.H, k.W, k.weight);
    load_1d_array(bfile, k.bias);
    int out_block = (k.C_out + 1) / 4;
    int in_block = (k.C_in + 1) / 4;
    k.weight_buf = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * out_block * in_block * k.H * k.W * 4 * 4,
                                  k.weight.data(),
                                  NULL);
    k.bias_buf = clCreateBuffer(context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * out_block * 4,
                                k.bias.data(),
                                NULL);
}

void release_kernel(kernel &k) {
    clReleaseMemObject(k.weight_buf);
    clReleaseMemObject(k.bias_buf);
}

void load_norm_stat(char* mfile, char* vfile, norm_stat &n, cl_context context) {
    load_1d_array(mfile, n.mean);
    load_1d_array(vfile, n.variance);
    int out_block = n.C / 4;
    n.mean_buf = clCreateBuffer(context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * out_block * 4,
                                n.mean.data(),
                                NULL);
    n.variance_buf = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * out_block * 4,
                                    n.variance.data(),
                                    NULL);
}

void release_norm_stat(norm_stat &n) {
    clReleaseMemObject(n.mean_buf);
    clReleaseMemObject(n.variance_buf);
}

void load_res_block(char* wfile1, char* bfile1, char* mfile1, char *vfile1,
    char* wfile2, char* bfile2, char* mfile2, char *vfile2, res_block &r, cl_context context) {

    load_kernel(wfile1, bfile1, r.r_1, context);
    load_norm_stat(mfile1, vfile1, r.r_2, context);
    load_kernel(wfile2, bfile2, r.r_5, context);
    load_norm_stat(mfile2, vfile2, r.r_6, context);
}

void release_res_block(res_block &r) {
    release_kernel(r.r_1);
    release_norm_stat(r.r_2);
    release_kernel(r.r_5);
    release_norm_stat(r.r_6);
}

void load_model(model &cycleGAN, cl_context context) {
    load_kernel((char*)"./pretrain/model.1.weight.txt", (char*)"./pretrain/model.1.bias.txt", cycleGAN.m_1, context);
    load_norm_stat((char*)"./pretrain/model.2.running_mean.txt", (char*)"./pretrain/model.2.running_var.txt", cycleGAN.m_2, context);
    load_kernel((char*)"./pretrain/model.4.weight.txt", (char*)"./pretrain/model.4.bias.txt", cycleGAN.m_4, context);
    load_norm_stat((char*)"./pretrain/model.5.running_mean.txt", (char*)"./pretrain/model.5.running_var.txt", cycleGAN.m_5, context);
    load_kernel((char*)"./pretrain/model.7.weight.txt", (char*)"./pretrain/model.7.bias.txt", cycleGAN.m_7, context);
    load_norm_stat((char*)"./pretrain/model.8.running_mean.txt", (char*)"./pretrain/model.8.running_var.txt", cycleGAN.m_8, context);
    load_res_block((char*)"./pretrain/model.10.conv_block.1.weight.txt", (char*)"./pretrain/model.10.conv_block.1.bias.txt",
        (char*)"./pretrain/model.10.conv_block.2.running_mean.txt", (char*)"./pretrain/model.10.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.10.conv_block.5.weight.txt", (char*)"./pretrain/model.10.conv_block.5.bias.txt",
        (char*)"./pretrain/model.10.conv_block.6.running_mean.txt", (char*)"./pretrain/model.10.conv_block.6.running_var.txt",
        cycleGAN.m_10, context);
    load_res_block((char*)"./pretrain/model.11.conv_block.1.weight.txt", (char*)"./pretrain/model.11.conv_block.1.bias.txt",
        (char*)"./pretrain/model.11.conv_block.2.running_mean.txt", (char*)"./pretrain/model.11.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.11.conv_block.5.weight.txt", (char*)"./pretrain/model.11.conv_block.5.bias.txt",
        (char*)"./pretrain/model.11.conv_block.6.running_mean.txt", (char*)"./pretrain/model.11.conv_block.6.running_var.txt",
        cycleGAN.m_11, context);
    load_res_block((char*)"./pretrain/model.12.conv_block.1.weight.txt", (char*)"./pretrain/model.12.conv_block.1.bias.txt",
        (char*)"./pretrain/model.12.conv_block.2.running_mean.txt", (char*)"./pretrain/model.12.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.12.conv_block.5.weight.txt", (char*)"./pretrain/model.12.conv_block.5.bias.txt",
        (char*)"./pretrain/model.12.conv_block.6.running_mean.txt", (char*)"./pretrain/model.12.conv_block.6.running_var.txt",
        cycleGAN.m_12, context);
    load_res_block((char*)"./pretrain/model.13.conv_block.1.weight.txt", (char*)"./pretrain/model.13.conv_block.1.bias.txt",
        (char*)"./pretrain/model.13.conv_block.2.running_mean.txt", (char*)"./pretrain/model.13.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.13.conv_block.5.weight.txt", (char*)"./pretrain/model.13.conv_block.5.bias.txt",
        (char*)"./pretrain/model.13.conv_block.6.running_mean.txt", (char*)"./pretrain/model.13.conv_block.6.running_var.txt",
        cycleGAN.m_13, context);
    load_res_block((char*)"./pretrain/model.14.conv_block.1.weight.txt", (char*)"./pretrain/model.14.conv_block.1.bias.txt",
        (char*)"./pretrain/model.14.conv_block.2.running_mean.txt", (char*)"./pretrain/model.14.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.14.conv_block.5.weight.txt", (char*)"./pretrain/model.14.conv_block.5.bias.txt",
        (char*)"./pretrain/model.14.conv_block.6.running_mean.txt", (char*)"./pretrain/model.14.conv_block.6.running_var.txt",
        cycleGAN.m_14, context);
    load_res_block((char*)"./pretrain/model.15.conv_block.1.weight.txt", (char*)"./pretrain/model.15.conv_block.1.bias.txt",
        (char*)"./pretrain/model.15.conv_block.2.running_mean.txt", (char*)"./pretrain/model.15.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.15.conv_block.5.weight.txt", (char*)"./pretrain/model.15.conv_block.5.bias.txt",
        (char*)"./pretrain/model.15.conv_block.6.running_mean.txt", (char*)"./pretrain/model.15.conv_block.6.running_var.txt",
        cycleGAN.m_15, context);
    load_res_block((char*)"./pretrain/model.16.conv_block.1.weight.txt", (char*)"./pretrain/model.16.conv_block.1.bias.txt",
        (char*)"./pretrain/model.16.conv_block.2.running_mean.txt", (char*)"./pretrain/model.16.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.16.conv_block.5.weight.txt", (char*)"./pretrain/model.16.conv_block.5.bias.txt",
        (char*)"./pretrain/model.16.conv_block.6.running_mean.txt", (char*)"./pretrain/model.16.conv_block.6.running_var.txt",
        cycleGAN.m_16, context);
    load_res_block((char*)"./pretrain/model.17.conv_block.1.weight.txt", (char*)"./pretrain/model.17.conv_block.1.bias.txt",
        (char*)"./pretrain/model.17.conv_block.2.running_mean.txt", (char*)"./pretrain/model.17.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.17.conv_block.5.weight.txt", (char*)"./pretrain/model.17.conv_block.5.bias.txt",
        (char*)"./pretrain/model.17.conv_block.6.running_mean.txt", (char*)"./pretrain/model.17.conv_block.6.running_var.txt",
        cycleGAN.m_17, context);
    load_res_block((char*)"./pretrain/model.18.conv_block.1.weight.txt", (char*)"./pretrain/model.18.conv_block.1.bias.txt",
        (char*)"./pretrain/model.18.conv_block.2.running_mean.txt", (char*)"./pretrain/model.18.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.18.conv_block.5.weight.txt", (char*)"./pretrain/model.18.conv_block.5.bias.txt",
        (char*)"./pretrain/model.18.conv_block.6.running_mean.txt", (char*)"./pretrain/model.18.conv_block.6.running_var.txt",
        cycleGAN.m_18, context);
    load_kernel((char*)"./pretrain/model.19.weight.txt", (char*)"./pretrain/model.19.bias.txt", cycleGAN.m_19, context);
    load_norm_stat((char*)"./pretrain/model.20.running_mean.txt", (char*)"./pretrain/model.20.running_var.txt", cycleGAN.m_20, context);
    load_kernel((char*)"./pretrain/model.22.weight.txt", (char*)"./pretrain/model.22.bias.txt", cycleGAN.m_22, context);
    load_norm_stat((char*)"./pretrain/model.23.running_mean.txt", (char*)"./pretrain/model.23.running_var.txt", cycleGAN.m_23, context);
    load_kernel((char*)"./pretrain/model.26.weight.txt", (char*)"./pretrain/model.26.bias.txt", cycleGAN.m_26, context);
}

void release_model(model &cycleGAN) {
    release_kernel(cycleGAN.m_1);
    release_norm_stat(cycleGAN.m_2);
    release_kernel(cycleGAN.m_4);
    release_norm_stat(cycleGAN.m_5);
    release_kernel(cycleGAN.m_7);
    release_norm_stat(cycleGAN.m_8);
    release_res_block(cycleGAN.m_10);
    release_res_block(cycleGAN.m_11);
    release_res_block(cycleGAN.m_12);
    release_res_block(cycleGAN.m_13);
    release_res_block(cycleGAN.m_14);
    release_res_block(cycleGAN.m_15);
    release_res_block(cycleGAN.m_16);
    release_res_block(cycleGAN.m_17);
    release_res_block(cycleGAN.m_18);
    release_kernel(cycleGAN.m_19);
    release_norm_stat(cycleGAN.m_20);
    release_kernel(cycleGAN.m_22);
    release_norm_stat(cycleGAN.m_23);
    release_kernel(cycleGAN.m_26);
}

}