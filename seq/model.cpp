#include "seq/model.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_1d_array(const char* filepath, std::vector<float> &data) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(filepath, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1) {
        char *token, *str, *tofree, *ptr;
        tofree = str = strdup(line);
        data.push_back(strtof(str, &ptr));
        free(tofree);
    }

    fclose(fp);
    if (line)
        free(line);
}

void load_4d_array(const char* filepath, int N, int C, int H, int W, Tensor4D &data) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int line_count = 0;

    fp = fopen(filepath, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1) {
        int n = line_count / (C * H);
        int c = (line_count % (C * H)) / H;
        int h = line_count % (C * H) % H;
        int w = 0;
        char *token, *str, *tofree;
        tofree = str = strdup(line);
        while ((token = strsep(&str, " "))) {
            char *ptr;
            data(n, c, h, w) = strtof(token, &ptr);
            w++;
        }
        free(tofree);
        line_count++;
    }
    fclose(fp);
    if (line)
        free(line);
}

void load_kernel(char* wfile, char* bfile, kernel &k) {
    load_4d_array(wfile, k.C_out, k.C, k.H, k.W, k.weight);
    load_1d_array(bfile, k.bias);
}

void load_norm_stat(char* mfile, char* vfile, norm_stat &n) {
    load_1d_array(mfile, n.mean);
    load_1d_array(vfile, n.variance);
}

void load_res_block(char* wfile1, char* bfile1, char* mfile1, char *vfile1,
    char* wfile2, char* bfile2, char* mfile2, char *vfile2, res_block &r) {

    load_kernel(wfile1, bfile1, r.r_1);
    load_norm_stat(mfile1, vfile1, r.r_2);
    load_kernel(wfile2, bfile2, r.r_5);
    load_norm_stat(mfile2, vfile2, r.r_6);
}

void load_model(model &cycleGAN) {
    load_kernel((char*)"./pretrain/model.1.weight.txt", (char*)"./pretrain/model.1.bias.txt", cycleGAN.m_1);
    load_norm_stat((char*)"./pretrain/model.2.running_mean.txt", (char*)"./pretrain/model.2.running_var.txt", cycleGAN.m_2);
    load_kernel((char*)"./pretrain/model.4.weight.txt", (char*)"./pretrain/model.4.bias.txt", cycleGAN.m_4);
    load_norm_stat((char*)"./pretrain/model.5.running_mean.txt", (char*)"./pretrain/model.5.running_var.txt", cycleGAN.m_5);
    load_kernel((char*)"./pretrain/model.7.weight.txt", (char*)"./pretrain/model.7.bias.txt", cycleGAN.m_7);
    load_norm_stat((char*)"./pretrain/model.8.running_mean.txt", (char*)"./pretrain/model.8.running_var.txt", cycleGAN.m_8);
    load_res_block((char*)"./pretrain/model.10.conv_block.1.weight.txt", (char*)"./pretrain/model.10.conv_block.1.bias.txt",
        (char*)"./pretrain/model.10.conv_block.2.running_mean.txt", (char*)"./pretrain/model.10.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.10.conv_block.5.weight.txt", (char*)"./pretrain/model.10.conv_block.5.bias.txt",
        (char*)"./pretrain/model.10.conv_block.6.running_mean.txt", (char*)"./pretrain/model.10.conv_block.6.running_var.txt",
        cycleGAN.m_10);
    load_res_block((char*)"./pretrain/model.11.conv_block.1.weight.txt", (char*)"./pretrain/model.11.conv_block.1.bias.txt",
        (char*)"./pretrain/model.11.conv_block.2.running_mean.txt", (char*)"./pretrain/model.11.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.11.conv_block.5.weight.txt", (char*)"./pretrain/model.11.conv_block.5.bias.txt",
        (char*)"./pretrain/model.11.conv_block.6.running_mean.txt", (char*)"./pretrain/model.11.conv_block.6.running_var.txt",
        cycleGAN.m_11);
    load_res_block((char*)"./pretrain/model.12.conv_block.1.weight.txt", (char*)"./pretrain/model.12.conv_block.1.bias.txt",
        (char*)"./pretrain/model.12.conv_block.2.running_mean.txt", (char*)"./pretrain/model.12.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.12.conv_block.5.weight.txt", (char*)"./pretrain/model.12.conv_block.5.bias.txt",
        (char*)"./pretrain/model.12.conv_block.6.running_mean.txt", (char*)"./pretrain/model.12.conv_block.6.running_var.txt",
        cycleGAN.m_12);
    load_res_block((char*)"./pretrain/model.13.conv_block.1.weight.txt", (char*)"./pretrain/model.13.conv_block.1.bias.txt",
        (char*)"./pretrain/model.13.conv_block.2.running_mean.txt", (char*)"./pretrain/model.13.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.13.conv_block.5.weight.txt", (char*)"./pretrain/model.13.conv_block.5.bias.txt",
        (char*)"./pretrain/model.13.conv_block.6.running_mean.txt", (char*)"./pretrain/model.13.conv_block.6.running_var.txt",
        cycleGAN.m_13);
    load_res_block((char*)"./pretrain/model.14.conv_block.1.weight.txt", (char*)"./pretrain/model.14.conv_block.1.bias.txt",
        (char*)"./pretrain/model.14.conv_block.2.running_mean.txt", (char*)"./pretrain/model.14.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.14.conv_block.5.weight.txt", (char*)"./pretrain/model.14.conv_block.5.bias.txt",
        (char*)"./pretrain/model.14.conv_block.6.running_mean.txt", (char*)"./pretrain/model.14.conv_block.6.running_var.txt",
        cycleGAN.m_14);
    load_res_block((char*)"./pretrain/model.15.conv_block.1.weight.txt", (char*)"./pretrain/model.15.conv_block.1.bias.txt",
        (char*)"./pretrain/model.15.conv_block.2.running_mean.txt", (char*)"./pretrain/model.15.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.15.conv_block.5.weight.txt", (char*)"./pretrain/model.15.conv_block.5.bias.txt",
        (char*)"./pretrain/model.15.conv_block.6.running_mean.txt", (char*)"./pretrain/model.15.conv_block.6.running_var.txt",
        cycleGAN.m_15);
    load_res_block((char*)"./pretrain/model.16.conv_block.1.weight.txt", (char*)"./pretrain/model.16.conv_block.1.bias.txt",
        (char*)"./pretrain/model.16.conv_block.2.running_mean.txt", (char*)"./pretrain/model.16.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.16.conv_block.5.weight.txt", (char*)"./pretrain/model.16.conv_block.5.bias.txt",
        (char*)"./pretrain/model.16.conv_block.6.running_mean.txt", (char*)"./pretrain/model.16.conv_block.6.running_var.txt",
        cycleGAN.m_16);
    load_res_block((char*)"./pretrain/model.17.conv_block.1.weight.txt", (char*)"./pretrain/model.17.conv_block.1.bias.txt",
        (char*)"./pretrain/model.17.conv_block.2.running_mean.txt", (char*)"./pretrain/model.17.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.17.conv_block.5.weight.txt", (char*)"./pretrain/model.17.conv_block.5.bias.txt",
        (char*)"./pretrain/model.17.conv_block.6.running_mean.txt", (char*)"./pretrain/model.17.conv_block.6.running_var.txt",
        cycleGAN.m_17);
    load_res_block((char*)"./pretrain/model.18.conv_block.1.weight.txt", (char*)"./pretrain/model.18.conv_block.1.bias.txt",
        (char*)"./pretrain/model.18.conv_block.2.running_mean.txt", (char*)"./pretrain/model.18.conv_block.2.running_var.txt",
        (char*)"./pretrain/model.18.conv_block.5.weight.txt", (char*)"./pretrain/model.18.conv_block.5.bias.txt",
        (char*)"./pretrain/model.18.conv_block.6.running_mean.txt", (char*)"./pretrain/model.18.conv_block.6.running_var.txt",
        cycleGAN.m_18);
    load_kernel((char*)"./pretrain/model.19.weight.txt", (char*)"./pretrain/model.19.bias.txt", cycleGAN.m_19);
    load_norm_stat((char*)"./pretrain/model.20.running_mean.txt", (char*)"./pretrain/model.20.running_var.txt", cycleGAN.m_20);
    load_kernel((char*)"./pretrain/model.22.weight.txt", (char*)"./pretrain/model.22.bias.txt", cycleGAN.m_22);
    load_norm_stat((char*)"./pretrain/model.23.running_mean.txt", (char*)"./pretrain/model.23.running_var.txt", cycleGAN.m_23);
    load_kernel((char*)"./pretrain/model.26.weight.txt", (char*)"./pretrain/model.26.bias.txt", cycleGAN.m_26);
}