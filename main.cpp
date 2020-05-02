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

#include <CL/cl.h>

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext() {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) firstPlatformId,
            0
        };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Failed to create an OpenCL GPU or CPU context."
                      << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              0,
                              NULL,
                              &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        std::cerr
            << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceBufferSize,
                              devices,
                              NULL);
    if (errNum != CL_SUCCESS) {
        delete[] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL) {
        delete[] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context,
                         cl_device_id device,
                         const char *fileName) {
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file for reading: " << fileName
                  << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) &srcStr,
                                        NULL, NULL);
    if (program == NULL) {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b) {
    memObjects[0] =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL
        || memObjects[2] == NULL) {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

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
    program = CreateProgram(context, device, "conv2d.cl");
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
