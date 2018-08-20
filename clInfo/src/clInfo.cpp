#include <iostream>
#include <vector>
#include <string>
#include "CL/cl.h"

#define checkError(err) \
  if(CL_SUCCESS != err) { \
    printf("line %d : error code %d\n", __LINE__, err); \
  }

using namespace std;
int main () {
  std::cout << "clInfomation begin" << std::endl;
  std::vector<std::pair<std::string, cl_platform_info>> platform_info_str={
    {"CL_PLATFORM_PROFILE", CL_PLATFORM_PROFILE},
    {"CL_PLATFORM_VERSION", CL_PLATFORM_VERSION},
    {"CL_PLATFORM_NAME", CL_PLATFORM_NAME},
    {"CL_PLATFORM_VENDOR", CL_PLATFORM_VENDOR},
    {"CL_PLATFORM_EXTENSIONS", CL_PLATFORM_EXTENSIONS}};

  std::vector<std::pair<std::string, cl_device_info>> device_info_str={
    {"CL_DEVICE_NAME", CL_DEVICE_NAME},
    {"CL_DEVICE_PROFILE", CL_DEVICE_PROFILE},
    {"CL_DEVICE_EXTENSIONS", CL_DEVICE_EXTENSIONS},
    {"CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION},
    {"CL_DEVICE_VENDOR", CL_DEVICE_VENDOR},
    {"CL_DEVICE_VERSION", CL_DEVICE_VERSION},
    {"CL_DRIVER_VERSION", CL_DRIVER_VERSION}};

  std::vector<std::pair<std::string, cl_device_info>> device_info_uint={
    {"CL_DEVICE_ADDRESS_BITS", CL_DEVICE_ADDRESS_BITS},
    {"CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE", CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE},
    {"CL_DEVICE_MAX_CLOCK_FREQUENCY", CL_DEVICE_MAX_CLOCK_FREQUENCY},
    {"CL_DEVICE_MAX_COMPUTE_UNITS", CL_DEVICE_MAX_COMPUTE_UNITS},
    {"CL_DEVICE_MAX_CONSTANT_ARGS", CL_DEVICE_MAX_CONSTANT_ARGS},
    {"CL_DEVICE_MAX_READ_IMAGE_ARGS", CL_DEVICE_MAX_READ_IMAGE_ARGS},
    {"CL_DEVICE_MAX_SAMPLERS", CL_DEVICE_MAX_SAMPLERS},
    {"CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS},
    {"CL_DEVICE_MAX_WRITE_IMAGE_ARGS", CL_DEVICE_MAX_WRITE_IMAGE_ARGS},
    {"CL_DEVICE_MEM_BASE_ADDR_ALIGN", CL_DEVICE_MEM_BASE_ADDR_ALIGN},
    {"CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE", CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE},
    {"CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR", CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR},
    {"CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT", CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT},
    {"CL_DEVICE_NATIVE_VECTOR_WIDTH_INT", CL_DEVICE_NATIVE_VECTOR_WIDTH_INT},
    {"CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG", CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG},
    {"CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT", CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT},
    {"CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE", CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE},
    {"CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF", CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF},
    {"CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR},
    {"CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT},
    {"CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT},
    {"CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG", CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG},
    {"CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT},
    {"CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE", CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE},
    {"CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF", CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF},
    {"CL_DEVICE_VENDOR_ID", CL_DEVICE_VENDOR_ID}};

  std::vector<std::pair<std::string, cl_device_info>> device_info_bool={
    {"CL_DEVICE_AVAILABLE", CL_DEVICE_AVAILABLE},
    {"CL_DEVICE_COMPILER_AVAILABLE", CL_DEVICE_COMPILER_AVAILABLE},
    {"CL_DEVICE_ENDIAN_LITTLE", CL_DEVICE_ENDIAN_LITTLE},
    {"CL_DEVICE_ERROR_CORRECTION_SUPPORT", CL_DEVICE_ERROR_CORRECTION_SUPPORT},
    {"CL_DEVICE_HOST_UNIFIED_MEMORY", CL_DEVICE_HOST_UNIFIED_MEMORY},
    {"CL_DEVICE_IMAGE_SUPPORT", CL_DEVICE_IMAGE_SUPPORT}};

  std::vector<std::pair<std::string, cl_device_info>> device_info_ulong={
    {"CL_DEVICE_GLOBAL_MEM_CACHE_SIZE", CL_DEVICE_GLOBAL_MEM_CACHE_SIZE},
    {"CL_DEVICE_GLOBAL_MEM_SIZE", CL_DEVICE_GLOBAL_MEM_SIZE},
    {"CL_DEVICE_LOCAL_MEM_SIZE", CL_DEVICE_LOCAL_MEM_SIZE},
    {"CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE", CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE},
    {"CL_DEVICE_MAX_MEM_ALLOC_SIZE", CL_DEVICE_MAX_MEM_ALLOC_SIZE}};

  std::vector<std::pair<std::string, cl_device_info>> device_info_size_t={
    {"CL_DEVICE_IMAGE2D_MAX_HEIGHT", CL_DEVICE_IMAGE2D_MAX_HEIGHT},
    {"CL_DEVICE_IMAGE2D_MAX_WIDTH", CL_DEVICE_IMAGE2D_MAX_WIDTH},
    {"CL_DEVICE_IMAGE3D_MAX_DEPTH", CL_DEVICE_IMAGE3D_MAX_DEPTH},
    {"CL_DEVICE_IMAGE3D_MAX_HEIGHT", CL_DEVICE_IMAGE3D_MAX_HEIGHT},
    {"CL_DEVICE_IMAGE3D_MAX_WIDTH", CL_DEVICE_IMAGE3D_MAX_WIDTH},
    {"CL_DEVICE_MAX_PARAMETER_SIZE", CL_DEVICE_MAX_PARAMETER_SIZE},
    {"CL_DEVICE_MAX_WORK_GROUP_SIZE", CL_DEVICE_MAX_WORK_GROUP_SIZE},
    {"CL_DEVICE_PROFILING_TIMER_RESOLUTION", CL_DEVICE_PROFILING_TIMER_RESOLUTION}};

  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  checkError(clGetPlatformIDs(1, &platform, nullptr));
  checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  checkError(err);

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  std::cout << "platform : " << platform << std::endl;
  std::cout << "device : " << device << std::endl;
  std::cout << "context : " << context << std::endl;

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  for(const auto & item : platform_info_str) {
    vector<char> data(10240);
    size_t ret_size;
    err = clGetPlatformInfo(platform, item.second,
        data.size() * sizeof(char), data.data(), &ret_size);
    data.resize(ret_size);
    if(CL_SUCCESS == err) {
      std::cout << data.data() << std::endl;
    }
    else {
      std::cout << "not available" << std::endl;
    }
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  for(const auto & item : device_info_str) {
    vector<char> data(10240);
    size_t ret_size;
    err = clGetDeviceInfo(device, item.second,
        data.size() * sizeof(char), data.data(), &ret_size);
    data.resize(ret_size);
    if(CL_SUCCESS == err) {
      std::cout << data.data() << std::endl;
    }
    else {
      std::cout << "not available" << std::endl;
    }
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  for(const auto & item : device_info_uint) {
    cl_uint data;
    size_t ret_size;
    std::cout << item.first << " : ";
    err = clGetDeviceInfo(device, item.second,
        sizeof(data), &data, &ret_size);
    if(CL_SUCCESS == err) {
      std::cout << data << std::endl;
    }
    else {
      std::cout << "not available" << std::endl;
    }
  }
  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  for(const auto & item : device_info_bool) {
    cl_bool data;
    size_t ret_size;
    std::cout << item.first << " : ";
    err = clGetDeviceInfo(device, item.second,
        sizeof(data), &data, &ret_size);
    if(CL_FALSE == data) {
      std::cout << "CL_FALSE";
    }
    else if(CL_TRUE == data){
      std::cout << "CL_TRUE";
    }
    else if(CL_BLOCKING == data) {
      std::cout << "CL_BLOCKING";
    }
    else if(CL_NON_BLOCKING == data) {
      std::cout << "CL_NON_BLOCKING";
    }
    else {
      std::cout << "not available";
    }
    std::cout << std::endl;
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  for(const auto & item : device_info_ulong) {
    cl_ulong data;
    size_t ret_size;
    std::cout << item.first << " : ";
    err = clGetDeviceInfo(device, item.second,
          sizeof(data), &data, &ret_size);
    if(CL_SUCCESS == err) {
      std::cout << data;
    }
    else {
      std::cout << "not available";
    }
    std::cout << std::endl;
  }


  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  for(const auto & item : device_info_size_t) {
    size_t data;
    size_t ret_size;
    std::cout << item.first << " : ";
    err = clGetDeviceInfo(device, item.second,
        sizeof(data), &data, &ret_size);
    if(CL_SUCCESS == err) {
      std::cout << data;
    }
    else {
      std::cout << "not available";
    }
    std::cout << std::endl;
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;
  size_t ret_size;
  {
    cl_device_type data;
    std::cout << "CL_DEVICE_TYPE" << ": ";
    if(clGetDeviceInfo(device, CL_DEVICE_TYPE,
          sizeof(data), &data, &ret_size)>=0) {
      if((data & CL_DEVICE_TYPE_DEFAULT)!=0) {
        std::cout << "CL_DEVICE_TYPE_DEFAULT ";
      }
      if((data & CL_DEVICE_TYPE_CPU)!=0) {
        std::cout << "CL_DEVICE_TYPE_CPU ";
      }
      if((data & CL_DEVICE_TYPE_GPU)!=0) {
        std::cout << "CL_DEVICE_TYPE_GPU ";
      }
      if((data & CL_DEVICE_TYPE_ACCELERATOR)!=0) {
        std::cout << "CL_DEVICE_TYPE_ACCELERATOR ";
      }
      if((data & CL_DEVICE_TYPE_CUSTOM)!=0) {
        std::cout << "CL_DEVICE_TYPE_CUSTOM ";
      }
    }
    else {
      std::cout << "not available";
    }
    std::cout << endl;
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  {
    cl_device_mem_cache_type data;
    std::cout << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE" << ": ";
    if(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
          sizeof(data), &data, &ret_size)>=0) {
      if(data==CL_NONE) {
        std::cout << "CL_NONE";
      }
      else if(data==CL_READ_ONLY_CACHE) {
        std::cout << "CL_READ_ONLY_CACHE";
      }
      else if(data==CL_READ_WRITE_CACHE) {
        std::cout << "CL_READ_WRITE_CACHE";
      }
    }
    else {
      std::cout << "not available";
    }
    std::cout << endl;
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;
  {
    cl_device_local_mem_type data;
    std::cout << "CL_DEVICE_LOCAL_MEM_TYPE" << ": ";
    if(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE,
          sizeof(data), &data, &ret_size)>=0) {
      if(data==CL_LOCAL) {
        std::cout << "CL_LOCAL";
      }
      else if(data==CL_GLOBAL) {
        std::cout << "CL_GLOBAL";
      }
    }
    else {
      std::cout << "not available";
    }
    std::cout << endl;
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  {
    cl_command_queue_properties data;
    std::cout << "CL_DEVICE_QUEUE_PROPERTIES" << ": ";
    if(clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
          sizeof(data), &data, &ret_size)>=0) {
      if((data & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)!=0) {
        std::cout << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
      }
      if((data & CL_QUEUE_PROFILING_ENABLE)!=0) {
        std::cout << "CL_QUEUE_PROFILING_ENABLE ";
      }
    }
    else {
      std::cout << "not available";
    }
    std::cout << endl;
  }

  for(int i = 0; i < 20; ++i) std::cout << "-";
  std::cout << std::endl;

  {
    vector<size_t> data(1024);
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES" << ": ";
    if(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
          data.size()*sizeof(size_t), data.data(), &ret_size)>=0) {
      data.resize(ret_size/sizeof(size_t));
      for(size_t i=0; i<data.size(); i++) {
        std::cout << data[i] << " ";
      }
    }
    else {
      std::cout << "not available";
    }
    std::cout << endl;
  }


  return 0;
}
