#include <time.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "CL/cl.h"

cl_uint num_platforms;
cl_platform_id *platforms;
cl_uint num_devices;
cl_device_id *devices;
cl_context context;
cl_command_queue queue;
cl_int err;
cl_program program;
cl_kernel kernel;

#define LEN(arr) sizeof(arr) / sizeof(arr[0])
#define checkError(err)                                     \
    if (CL_SUCCESS != err)                                  \
    {                                                       \
        printf("line %d : error code %d\n", __LINE__, err); \
    }

void randomRangeData(float *data, int len, int maxNum = 255)
{
    clock_t ct = clock();
    srand((unsigned int)ct);

    for (int i = 0; i < len; ++i)
    {
        float tmp = rand() % maxNum;
        data[i] = (float)tmp / maxNum / 2.f;
    }
}

double uniGetTime()
{
    struct timespec cl_time;
    clock_gettime(CLOCK_MONOTONIC, &cl_time);
    double time = (double)cl_time.tv_sec + cl_time.tv_nsec * 1e-9;
    return time;
}

void oclInit()
{
    err = clGetPlatformIDs(0, 0, &num_platforms);
    checkError(err);
    platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    checkError(err);

    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    checkError(err);
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    checkError(err);

    context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
    checkError(err);
    queue = clCreateCommandQueue(context, devices[0], 0, &err);
    checkError(err);
}

const char *src_buffer_float[] = {
    "__kernel void ceilKernel(   \n"
    "__global const float* a,         \n"
    "__global float*       b,         \n"
    "int w,                           \n"
    "int h ) {                        \n"
    "   int idx = get_global_id(0);   \n"
    "   int idy = get_global_id(1);   \n"
    "   if(idx * 4 >= w || idy  >= h) return;   \n"
    "   int ix = idy * w + idx * 4; \n"
    "   __global const float* a_off = a + ix; \n"
    "   __global float* b_off = b + ix; \n"
    "   float4 val = vload4(0, a_off);  \n"
    "   float4 res = val;               \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   res += 1.00f * res;              \n"
    "   res += 0.01f * res;              \n"
    "   res += 0.02f * res;              \n"
    "   res += 0.03f * res;              \n"
    "   res += 0.04f * res;              \n"
    "   res += 0.05f * res;              \n"
    "   res += 0.06f * res;              \n"
    "   res += 0.07f * res;              \n"
    "   res += 0.08f * res;              \n"
    "   res += 0.09f * res;              \n"
    "                                   \n"
    "   vstore4(res, 0, b_off); \n"
    "}"};

const char *src_buffer_half[] = {
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable \n"
    "__kernel void ceilKernel(   \n"
    "__global const float* a,         \n"
    "__global float*       b,         \n"
    "int w,                           \n"
    "int h ) {                        \n"
    "   int idx = get_global_id(0);   \n"
    "   int idy = get_global_id(1);   \n"
    "   if(idx * 4 >= w || idy  >= h) return;   \n"
    "   int ix = idy * w + idx * 4; \n"
    "   __global const float* a_off = a + ix; \n"
    "   __global float* b_off = b + ix; \n"
    "   float4 val = vload4(0, a_off);  \n"
    "   half4 res = convert_half4(val);  \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                   \n"
    "   res += (half)1.00 * res;              \n"
    "   res += (half)0.01 * res;              \n"
    "   res += (half)0.02 * res;              \n"
    "   res += (half)0.03 * res;              \n"
    "   res += (half)0.04 * res;              \n"
    "   res += (half)0.05 * res;              \n"
    "   res += (half)0.06 * res;              \n"
    "   res += (half)0.07 * res;              \n"
    "   res += (half)0.08 * res;              \n"
    "   res += (half)0.09 * res;              \n"
    "                                    \n"
    "   float4 out = convert_float4(res);   \n"
    "   vstore4(out, 0, b_off); \n"
    "}"};

double halfPeak(cl_mem &a_dev, cl_mem &b_dev, int num)
{
    cl_program program_half = clCreateProgramWithSource(context, LEN(src_buffer_half), src_buffer_half, NULL, NULL);
    cl_int err;
    err = clBuildProgram(program_half, num_devices, devices, NULL, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        size_t log_size;
        checkError(clGetProgramBuildInfo(program_half, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
        std::vector<char> error_log(log_size);
        printf("log_size is : %d", log_size);
        checkError(clGetProgramBuildInfo(program_half, devices[0], CL_PROGRAM_BUILD_LOG, log_size, error_log.data(), NULL))
            printf("%s \n", error_log.data());
        exit(-1);
    }
    cl_kernel kernel_buffer = clCreateKernel(program_half, "ceilKernel", NULL);
    checkError(clSetKernelArg(kernel_buffer, 0, sizeof(cl_mem), &a_dev));
    checkError(clSetKernelArg(kernel_buffer, 1, sizeof(cl_mem), &b_dev));
    checkError(clSetKernelArg(kernel_buffer, 2, sizeof(int), &num));
    checkError(clSetKernelArg(kernel_buffer, 3, sizeof(int), &num));

    const size_t globalWorkSize[2] = {static_cast<size_t>(num / 4), static_cast<size_t>(num)};
    double start = uniGetTime();
    int iter = 100;
    for (int i = 0; i < iter; ++i)
    {
        checkError(clEnqueueNDRangeKernel(queue, kernel_buffer, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL));
    }
    clFinish(queue);
    double end = uniGetTime();
    double time_half = end - start;
    return time_half / iter;
}

void cpuCompute(float *a_host, float *check_host, int num)
{
    for (int i = 0; i < num * num; ++i)
    {
        float val = a_host[i];
        float res = val;
        for (int k = 0; k < 10; ++k)
            for (int j = 0; j < 10; ++j)
            {
                if (0 == j)
                {
                    res += (1 + j * 0.01) * res;
                }
                else
                {
                    res += (j * 0.01) * res;
                }
            }
        check_host[i] = res;
    }
}
void checkHalf(float *check_host, float *opencl_res, int num)
{
    bool flag = true;
    for (int i = 0; i < num * num; ++i)
    {
        if (opencl_res[i] > 65535)
        {
            printf("%d %f\n", i, opencl_res[i]);
            break;
        }
        if ((fabs(opencl_res[i] - check_host[i]) / check_host[i]) > 2 * 1e-2)
        {
            printf("opencl %f cpu %f diff %f\n", opencl_res[i], check_host[i],
                   fabs(opencl_res[i] - check_host[i]) / check_host[i]);
            flag = false;
            break;
        }
    }
    if (flag)
    {
        printf("half result is right\n");
    }
    else
    {
        printf("half result is wrong\n");
    }
}

double floatPeak(cl_mem &a_dev, cl_mem &b_dev, int num)
{
    cl_program program_float = clCreateProgramWithSource(context, LEN(src_buffer_float), src_buffer_float, NULL, NULL);
    err = clBuildProgram(program_float, num_devices, devices, NULL, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        size_t log_size;
        checkError(clGetProgramBuildInfo(program_float, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
        std::vector<char> error_log(log_size);
        printf("log_size is : %d", log_size);
        checkError(clGetProgramBuildInfo(program_float, devices[0], CL_PROGRAM_BUILD_LOG, log_size, error_log.data(), NULL))
            printf("%s \n", error_log.data());
        exit(-1);
    }
    cl_kernel kernel_float = clCreateKernel(program_float, "ceilKernel", NULL);
    checkError(clSetKernelArg(kernel_float, 0, sizeof(cl_mem), &a_dev));
    checkError(clSetKernelArg(kernel_float, 1, sizeof(cl_mem), &b_dev));
    checkError(clSetKernelArg(kernel_float, 2, sizeof(int), &num));
    checkError(clSetKernelArg(kernel_float, 3, sizeof(int), &num));

    const size_t globalWorkSize[2] = {static_cast<size_t>(num / 4), static_cast<size_t>(num)};
    double start = uniGetTime();
    int iter = 100;
    for (int i = 0; i < iter; ++i)
    {
        checkError(clEnqueueNDRangeKernel(queue, kernel_float, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL));
    }
    clFinish(queue);
    double end = uniGetTime();
    double time_float = end - start;
    return time_float / iter;
}

void checkFloat(float *check_host, float *opencl_res, int num)
{
    bool flag = true;
    for (int i = 0; i < num * num; ++i)
    {
        if ((fabs(opencl_res[i] - check_host[i]) / check_host[i]) > 1e-5)
        {
            printf("opencl %f cpu %f diff %f\n", opencl_res[i], check_host[i],
                   fabs(opencl_res[i] - check_host[i]) / check_host[i]);
            flag = false;
            break;
        }
    }
    if (flag)
    {
        printf("float result is right\n");
    }
    else
    {
        printf("float result is wrong\n");
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("./ocl_example num\n");
        return -1;
    }
    int num = atoi(argv[1]);
    num = (num + 3) / 4 * 4;
    printf("pad num %d\n", num);
    float *a_host = (float *)malloc(num * num * sizeof(float));
    float *b_host_half = (float *)malloc(num * num * sizeof(float));
    float *b_host_float = (float *)malloc(num * num * sizeof(float));
    float *check_host = (float *)malloc(num * num * sizeof(float));
    randomRangeData(a_host, num);

    oclInit();
    cl_mem a_buffer_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                         num * num * sizeof(float), NULL, &err);
    checkError(err);
    cl_mem b_buffer_dev0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          num * num * sizeof(float), NULL, &err);
    checkError(err);
    cl_mem b_buffer_dev1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                          num * num * sizeof(float), NULL, &err);
    checkError(err);

    double start = uniGetTime();
    checkError(clEnqueueWriteBuffer(queue, a_buffer_dev,
                                    CL_TRUE, 0, num * sizeof(float), a_host, 0, NULL, NULL));
    double end = uniGetTime();

    double op = static_cast<double>(num * num) * 2.f * 100.f / 1e9;
    double time_half = halfPeak(a_buffer_dev, b_buffer_dev0, num);
    double time_float = floatPeak(a_buffer_dev, b_buffer_dev1, num);
    printf("ops : %fG\n", op);
    printf("half costs %fs\t", time_half);
    printf("half ceil gflops : %f\n", op / time_half);
    printf("half costs %fs\t", time_float);
    printf("float ceil gflops : %f\n", op / time_float);

    checkError(clEnqueueReadBuffer(queue, b_buffer_dev0, CL_TRUE, 0,
                                   num * num * sizeof(float), b_host_half, 0, NULL, NULL));
    checkError(clEnqueueReadBuffer(queue, b_buffer_dev1, CL_TRUE, 0,
                                   num * num * sizeof(float), b_host_float, 0, NULL, NULL));

    cpuCompute(a_host, check_host, num);
    checkHalf(check_host, b_host_half, num);
    checkFloat(check_host, b_host_float, num);

    free(a_host);
    free(b_host_half);
    free(b_host_float);
    free(check_host);
    clReleaseMemObject(a_buffer_dev);
    clReleaseMemObject(b_buffer_dev0);
    clReleaseMemObject(b_buffer_dev1);
    free(platforms);
    free(devices);
    return 0;
}
