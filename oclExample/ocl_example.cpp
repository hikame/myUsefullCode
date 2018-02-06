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
    free(platforms);
    free(devices);
}

const char* src[] = {
    "__kernel void addKernel(   \n"
    "__global const float* a,         \n"
    "__global const float* b,         \n"
    "__global float*       c,         \n"
    "int len) {                       \n"
    "   int idx = get_global_id(0);   \n"
    "   if(idx >= len) return;        \n"
    "   c[idx] = a[idx] + b[idx];     \n"
    "}"
};

int main(int argc, char** argv) {
    int num = 1024;
    if(argc < 2) {
        printf("./ocl_example num\n");
        return -1;
    }
    num = atoi(argv[1]);
    float* a_host = (float*) malloc(num * sizeof(float));
    float* b_host = (float*) malloc(num * sizeof(float));
    float* c_host = (float*) malloc(num * sizeof(float));
    float* check_host = (float*) malloc(num * sizeof(float));
    randomRangeData(a_host, num);
    randomRangeData(b_host, num);

    oclInit();
    cl_mem a_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, num * sizeof(float), NULL, &err);
    checkError(err);
    cl_mem b_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, num * sizeof(float), NULL, &err);
    checkError(err);
    cl_mem c_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, num * sizeof(float), NULL, &err);
    checkError(err);

    double start = uniGetTime();
    checkError(clEnqueueWriteBuffer(queue, a_dev, CL_TRUE, 0, num * sizeof(float), a_host, 0, NULL, NULL));
    double end = uniGetTime();
    printf("A copy %d bytes from Host2Device costs %fs\n", num * sizeof(float), end - start);

    start = uniGetTime();
    checkError(clEnqueueWriteBuffer(queue, b_dev, CL_TRUE, 0, num * sizeof(float), b_host, 0, NULL, NULL));
    end = uniGetTime();
    printf("B copy %d bytes from Host2Device costs %fs\n", num * sizeof(float), end - start);

    program = clCreateProgramWithSource(context, LEN(src), src, NULL, NULL);
    checkError(clBuildProgram(program, num_devices, devices, NULL, NULL, NULL));
    kernel = clCreateKernel(program, "addKernel", NULL);
    checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_dev));
    checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_dev));
    checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_dev));
    checkError(clSetKernelArg(kernel, 3, sizeof(int), &num));

    const size_t globalWorkSize[1] = {num};
    start = uniGetTime();
    for(int i = 0; i < 20; ++i) {
        checkError(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL));
    }
    clFinish(queue);
    end = uniGetTime();
    printf("addKernel costs %fs\n", (end - start) / 20);

    start = uniGetTime();
    checkError(clEnqueueReadBuffer(queue, c_dev, CL_TRUE, 0, num * sizeof(float), c_host, 0, NULL, NULL));
    end = uniGetTime();
    printf("C copy %d bytes from Device2Host costs %fs\n", num * sizeof(float), end - start);

    start = uniGetTime();
    for(int i = 0; i < num; ++i) {
        check_host[i] = a_host[i] + b_host[i];
    }
    end = uniGetTime();
    printf("CPU costs %fs\n", end - start);

    bool flag = true;
    for(int i = 0; i < num; ++i) {
        if(check_host[i] - c_host[i] > 1e-3) {
            flag = false;
            break;
        }
        //printf("i %d : %f %f\n", i, check_host[i], c_host[i]);
    }
    if(flag) {
        printf("result is right\n");
    }

    free(a_host);
    free(b_host);
    free(c_host);
    free(check_host);
    clReleaseMemObject(a_dev);
    clReleaseMemObject(b_dev);
    clReleaseMemObject(c_dev);
    return 0;
}
