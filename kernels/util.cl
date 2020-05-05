// Copied from opencl programming guide

__kernel void add(__global float *src,
                  __global float *dst,
                  int len)
{
    int id = get_global_id(0);
    if (id < len)
        dst[id] += src[id];
}