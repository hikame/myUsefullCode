#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"

cl_uint             num_platforms;
cl_platform_id*     platforms;
cl_uint             num_devices;
cl_device_id*       devices;
cl_context          context;
cl_command_queue    queue;
cl_int              err;
cl_program          program;
cl_kernel           kernel;

#define LEN(arr) sizeof(arr) / sizeof(arr[0])
#define checkError(err) \
    if(CL_SUCCESS != err) { \
        printf("line %d : error code %d\n", __LINE__, err); \
    }

void randomRangeData(float* data, int len, int maxNum = 255) {
    clock_t ct = clock();
    srand((unsigned int)ct);

    for(int i = 0; i < len; ++i) {
        float tmp = rand() % maxNum;
        data[i] = (float) tmp;
    }
}

double uniGetTime() {
    struct timespec cl_time;
    clock_gettime(CLOCK_MONOTONIC, &cl_time);
    double time = (double)cl_time.tv_sec + cl_time.tv_nsec * 1e-9;
    return time;
}

void oclInit() {
    err = clGetPlatformIDs(0, 0, &num_platforms);
    checkError(err);
    platforms = (cl_platform_id*) malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    checkError(err);

    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    checkError(err);
    devices = (cl_device_id*) malloc(num_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    checkError(err);

    context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
    checkError(err);
    queue = clCreateCommandQueue(context, devices[0], 0, &err);
    checkError(err);
}

const char* src_buffer[] = {
    "__kernel void copyBufferKernel(   \n"
    "__global const float* a,         \n"
    "__global float*       b,         \n"
    "int w,                           \n"
    "int h) {                         \n"
    "   int idx = get_global_id(0);   \n"
    "   int idy = get_global_id(1);   \n"
    "   if(idx * 4 >= w || idy  >= h) return;   \n"
    "   int src_idx = idy * w + idx * 4; \n"
    "   __global const float* a_off = a + src_idx; \n"
    "   __global float* b_off = b + src_idx; \n"
    "   float4 val = vload4(0, a_off);  \n"
    "   //vstore4(val, 0, b_off); \n"
    "   //b_off[0] = a_off[0]; \n"
    "   //b_off[1] = a_off[1]; \n"
    "   //b_off[2] = a_off[2]; \n"
    "   //b_off[3] = a_off[3]; \n"
    "   b_off[0] = val.s0; \n"
    "   b_off[1] = val.s1; \n"
    "   b_off[2] = val.s2; \n"
    "   b_off[3] = val.s3; \n"
    "}"
};

const char* src_image[] = {
    "__kernel void copyImageKernel(   \n"
    "read_only image2d_t   a,         \n"
    "__global float*       b,         \n"
    "int w,                           \n"
    "int h) {                         \n"
    "   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | \n"
    "        CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; \n"
    "   int idx = get_global_id(0);   \n"
    "   int idy = get_global_id(1);   \n"
    "   if(idx * 4 >= w || idy >= h) return;   \n"
    "   int src_idx = idy * w + idx * 4; \n"
    "   int2 loc = {idx, idy}; \n"
    "   float4 val = read_imagef(a, smp, loc); \n"
    "   __global float* b_off = b + src_idx; \n"
    "   b_off[0] = val.s0; \n"
    "   b_off[1] = val.s1; \n"
    "   b_off[2] = val.s2; \n"
    "   b_off[3] = val.s3; \n"
    "}"
};

int main(int argc, char** argv) {
    if(argc < 2) {
        printf("./ocl_example num\n");
        return -1;
    }
    int num = atoi(argv[1]);
    printf("num is %d\n", num);

    cl_image_format = {CL_RGBA, CL_HALF_FLOAT};
    cl_mem a_image_dev = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        &format, num, num, 0, NULL, &err);
    checkError(err);

    return 0;
}
