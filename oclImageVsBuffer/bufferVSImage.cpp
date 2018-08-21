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
    num = (num + 7)/ 4 * 4;
    float* a_host = (float*) malloc(num * num * sizeof(float));
    float* b_host = (float*) malloc(num * num * sizeof(float));
    float* b_host2 = (float*) malloc(num * num * sizeof(float));
    randomRangeData(a_host, num);

    oclInit();
    cl_mem a_buffer_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
         num * num * sizeof(float), NULL, &err);
    checkError(err);
    cl_mem b_buffer_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
         num * num * sizeof(float), NULL, &err);
    checkError(err);

    cl_image_format format = {CL_RGBA, CL_FLOAT};
    size_t width = num / 4;
    size_t height = num;
    printf("width %d height %d\n", width, height);
    cl_mem a_image_dev = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        &format, width, height, 0, NULL, &err);
    checkError(err);
    cl_mem b_buffer_dev2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
         num * num * sizeof(float), NULL, &err);
    checkError(err);

    double start = uniGetTime();
    checkError(clEnqueueWriteBuffer(queue, a_buffer_dev,
        CL_TRUE, 0, num * sizeof(float), a_host, 0, NULL, NULL));
    double end = uniGetTime();

    size_t origin[] = {0, 0, 0};
    size_t region[] = {num / 4, num, 1};
    checkError(clEnqueueWriteImage(queue, a_image_dev, CL_TRUE,
        origin, region, 0, 0, a_host, 0, NULL, NULL));

    cl_program program_buffer = clCreateProgramWithSource(context, LEN(src_buffer), src_buffer, NULL, NULL);
    checkError(clBuildProgram(program_buffer, num_devices, devices, NULL, NULL, NULL));
    cl_kernel kernel_buffer = clCreateKernel(program_buffer, "copyBufferKernel", NULL);
    checkError(clSetKernelArg(kernel_buffer, 0, sizeof(cl_mem), &a_buffer_dev));
    checkError(clSetKernelArg(kernel_buffer, 1, sizeof(cl_mem), &b_buffer_dev));
    checkError(clSetKernelArg(kernel_buffer, 2, sizeof(int), &num));
    checkError(clSetKernelArg(kernel_buffer, 3, sizeof(int), &num));

    cl_program program_image = clCreateProgramWithSource(context, LEN(src_image), src_image, NULL, NULL);
    checkError(clBuildProgram(program_image, num_devices, devices, NULL, NULL, NULL));
    cl_kernel kernel_image = clCreateKernel(program_image, "copyImageKernel", NULL);
    checkError(clSetKernelArg(kernel_image, 0, sizeof(cl_mem), &a_image_dev));
    checkError(clSetKernelArg(kernel_image, 1, sizeof(cl_mem), &b_buffer_dev2));
    checkError(clSetKernelArg(kernel_image, 2, sizeof(int), &num));
    checkError(clSetKernelArg(kernel_image, 3, sizeof(int), &num));

    const size_t globalWorkSize[2] = {num / 4, num};
    start = uniGetTime();
    for(int i = 0; i < 20; ++i) {
        checkError(clEnqueueNDRangeKernel(queue, kernel_buffer, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL));
    }
    clFinish(queue);
    end = uniGetTime();
    double time_buffer = end - start;
    printf("copyBufferKernel costs %fs\n", time_buffer / 20);

    start = uniGetTime();
    for(int i = 0; i < 20; ++i) {
        checkError(clEnqueueNDRangeKernel(queue, kernel_image, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL));
    }
    clFinish(queue);
    end = uniGetTime();
    double time_image = end - start;
    printf("copyImageKernel costs %fs\n", time_image / 20);

    double diff_time = time_buffer - time_image;
    printf("image is faster than buffer : %f\n", diff_time / time_buffer);

    checkError(clEnqueueReadBuffer(queue, b_buffer_dev, CL_TRUE, 0,
         num * num * sizeof(float), b_host, 0, NULL, NULL));
    checkError(clEnqueueReadBuffer(queue, b_buffer_dev2, CL_TRUE, 0,
         num * num * sizeof(float), b_host2, 0, NULL, NULL));

    bool flag = true;
    for(int i = 0; i < num * num; ++i) {
        /*
        if(i < 20) {
            printf("buffer %f image %f\n", b_host[i], b_host2[i]);
        }
        */
        if(b_host2[i] - b_host[i] > 1e-3) {
            flag = false;
            break;
        }
    }
    if(flag) {
        printf("result is right\n");
    }
    else {
        printf("result is wrong\n");
    }

    free(a_host);
    free(b_host);
    free(b_host2);
    clReleaseMemObject(a_buffer_dev);
    clReleaseMemObject(b_buffer_dev);
    clReleaseMemObject(a_image_dev);
    clReleaseMemObject(b_buffer_dev2);
    free(platforms);
    free(devices);
    return 0;
}
