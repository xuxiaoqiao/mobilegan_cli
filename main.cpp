//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>

#include "parallel/runtime.hpp"

///
//  Constants
//
const int ARRAY_SIZE = 1000;


///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context,
             cl_command_queue commandQueue,
             cl_program program,
             cl_kernel kernel) {
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

///
//	main() for HelloWorld example
//
int main(int argc, char **argv) {
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = {0, 0, 0};
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL) {
        Cleanup(context, commandQueue, program, kernel);
        return 1;
    }

    // Create OpenCL program from conv2d.cl kernel source
    program = CreateProgram(context, device, "conv2d.cl", <#initializer#>);
    if (program == NULL) {
        Cleanup(context, commandQueue, program, kernel);
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "conv2d", NULL);
    if (kernel == NULL) {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    #define INPUT_CHANNEL_BLK 64
    #define INPUT_HW 66
    #define OUTPUT_CHANNEL_BLK 64
    #define OUTPUT_HW 64
    #define STRIDE 1
    float *input = new float[INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4];
    float *weight = new float[OUTPUT_CHANNEL_BLK * INPUT_CHANNEL_BLK * 3 * 3 * 4 * 4]; // C_out, C_in, Kh, Kw, 4c_out, 4c_in
    float *output = new float[OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4];

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
                        // weight[idx] = 1.0 / (9 * 4);

                    }
                }
            }
        }
    }

    cl_mem input_clbuf = clCreateBuffer(context,
                                        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        sizeof(float) * INPUT_CHANNEL_BLK * INPUT_HW * INPUT_HW * 4,
                                        input,
                                        NULL);
    cl_mem weight_clbuf = clCreateBuffer(context,
                                         CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         sizeof(float) * OUTPUT_CHANNEL_BLK * INPUT_CHANNEL_BLK * 3 * 3 * 4 * 4,
                                         weight,
                                         NULL);
    cl_mem output_clbuf = clCreateBuffer(context,
                                         CL_MEM_WRITE_ONLY,
                                         sizeof(float) * OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4,
                                         NULL,
                                         NULL);
    cl_mem bias_clbuf = nullptr;
    cl_int in_channel_num = INPUT_CHANNEL_BLK * 4;
    cl_int in_height = INPUT_HW;
    cl_int in_width = INPUT_HW;
    cl_int out_channel_num = OUTPUT_CHANNEL_BLK * 4;
    cl_int out_height = OUTPUT_HW;
    cl_int out_width = OUTPUT_HW;
    cl_int stride = STRIDE;
    cl_int kernel_height = 3;
    cl_int kernel_width = 3;

    if (input_clbuf == nullptr || weight_clbuf == nullptr || output_clbuf == nullptr) {
        std::cerr << "Error creating memory objects." << std::endl;
        return 1;
    }


    // if (!CreateMemObjects(context, memObjects, a, b))
    // {
    // Cleanup(context, commandQueue, program, kernel, memObjects);
    // return 1;
    // }

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_clbuf);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight_clbuf);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_clbuf);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_clbuf);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_int), &in_channel_num);
    errNum |= clSetKernelArg(kernel, 5, sizeof(cl_int), &in_height);
    errNum |= clSetKernelArg(kernel, 6, sizeof(cl_int), &in_width);
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_int), &out_channel_num);
    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_int), &out_height);
    errNum |= clSetKernelArg(kernel, 9, sizeof(cl_int), &out_width);
    errNum |= clSetKernelArg(kernel, 10, sizeof(cl_int), &stride);
    errNum |= clSetKernelArg(kernel, 11, sizeof(cl_int), &kernel_height);
    errNum |= clSetKernelArg(kernel, 12, sizeof(cl_int), &kernel_width);

    if (errNum != CL_SUCCESS) {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(context, commandQueue, program, kernel);
        return 1;
    }

    size_t globalWorkSize[3] = {OUTPUT_CHANNEL_BLK, OUTPUT_HW, OUTPUT_HW / 2}; // (C_block, H, W/UNROLL)
    size_t localWorkSize[3] = {1, 1};

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL,
                                    globalWorkSize, NULL,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel);
        return 1;
    }

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, output_clbuf, CL_TRUE,
                                 0, OUTPUT_CHANNEL_BLK * OUTPUT_HW * OUTPUT_HW * 4 * sizeof(float), output,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel);
        return 1;
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
    Cleanup(context, commandQueue, program, kernel);

    return 0;
}
