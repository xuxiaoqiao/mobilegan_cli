#ifndef PARALLEL_RUNTIME_HPP
#define PARALLEL_RUNTIME_HPP

//
// Some of the bootstrap code is copied from OpenCL Programming Guide
//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

#include <CL/cl.h>

cl_context CreateContext();
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
cl_program CreateProgram(cl_context context,
                         cl_device_id device,
                         const char *fileName,
                         const std::string &build_options);

#endif // PARALLEL_RUNTIME_HPP