#ifndef __OPENCL_INIT_HEAD_FILE__
#define __OPENCL_INIT_HEAD_FILE__
#include <map>
#include <iostream>
#include <vector>
#include "oclbin.h"
#include "CL/cl.h"

using namespace std;
#define uchar unsigned char

#define checkError(err) \
    if(CL_SUCCESS != err) { \
      std::cout << "file " << __FILE__ << " line " << __LINE__ << " err :" << err << std::endl; \
      exit(-1); \
    }

class OpenCL {
  public :
    OpenCL() {
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

    cl_context GetContext() {
      return context;
    }
    cl_command_queue GetCommandQueue() {
      return queue;
    }
    cl_device_id* GetDevices() {
      return devices;
    }

    ~OpenCL() {
      free(platforms);
      free(devices);
    }
  private :
    cl_uint             num_platforms;
    cl_platform_id*     platforms;
    cl_uint             num_devices;
    cl_device_id*       devices;
    cl_context          context;
    cl_command_queue    queue;
    cl_int              err;
    cl_program          program;
};

class KernelCache {
public:
    KernelCache() {}
    ~KernelCache() {
        for(auto kernel : kernelMap) {
            clReleaseKernel(kernel.second);
        }
        kernelMap.clear();
    }
    bool putKernel(const std::string key, cl_kernel kernel) {
        if(kernelMap.find(key) != kernelMap.end()) {
            kernelMap.insert(std::pair<std::string, cl_kernel>(key, kernel));
            return true;
        }
        else {
            return false;
        }
    }
    bool getKernel(const std::string key, cl_kernel* kernel) {
        if(kernelMap.find(key) != kernelMap.end()) {
            auto k = kernelMap.find(key);
            *kernel = k->second;
            return true;
        }
        else {
            *kernel = NULL;
            return false;
        }
    }

    bool FromKCGetKernel(cl_context context, string binName, string keyName, string kernelName, cl_kernel* kernel) {
        if(!getKernel(keyName, kernel)) {
            //if kernel is not in the kernelMap, we should find it from allOCLKernels.
            auto k = allOCLKernels.find(binName);
            if(k == allOCLKernels.end()) {
                fprintf(stderr, "%d : Nothing about %s kernel\n", __LINE__, binName.c_str());
                return false;
            }
            std::vector<cl_device_id> devices(1024);
            size_t ret_size;
            cl_int err, status;
            checkError(clGetContextInfo(context, CL_CONTEXT_DEVICES,
                             devices.size()*sizeof(cl_device_id),
                             devices.data(), &ret_size));
            devices.resize(ret_size / sizeof(cl_device_id));
            cl_program program;
            const uchar* data = k->second.bin;
            program = clCreateProgramWithBinary(context,
                                                1, &devices[0],
                                                &k->second.size,
                                                &data,
                                                &status, &err);
            checkError(err);
            const char* options = "";
            err = clBuildProgram(program, 1, &devices[0], options, NULL, NULL);
            checkError(err);
            if(CL_SUCCESS != err) {
                fprintf(stderr, "%s %d : build %s error\n", __FILE__, __LINE__, kernelName.c_str());
                return false;
            }
            *kernel = clCreateKernel(program, kernelName.c_str(), &err);
            checkError(err);
            putKernel(keyName, *kernel);
            clReleaseProgram(program);
        }
        return true;
    }

private:
    std::map<std::string, cl_kernel>kernelMap;
};
#endif
