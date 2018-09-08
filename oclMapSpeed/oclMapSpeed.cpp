#include "opencl_init.hpp"

int main(int argc, char** argv) {
  if(2 != argc) {
    std::cout << "./clMapInfo 1000 (Bytes)" << std::endl;
    exit(-1);
  }

  OpenCL aTest;
  cl_command_queue queue = aTest.GetCommandQueue();
  cl_context context = aTest.GetContext();

  int size = atoi(argv[1]);
  std::cout << "Alloc " << size << " bytes buffer." << std::endl;
  cl_int err;
  cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size,  nullptr, &err);
  checkError(err);

  cl_event event;
  vector<char>a_host(size, 1);
  vector<char>b_host(size);
  checkError(clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, size, a_host.data(), 0, nullptr, &event));

  checkError(clWaitForEvents(1, &event));
  clFinish(queue);

  cl_ulong time_end, time_start;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
    sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
    sizeof(time_end), &time_end, nullptr);
  double cost_time = time_end - time_start;
  for(int i = 0; i < 20; ++i) std::cout << "*";
  std::cout << std::endl;
  std::cout << "clEnqueueWriteBuffer : " << std::endl;

  std::cout << "cost time : " << cost_time  / 1e6 << " ms "<< std::endl;
  std::cout << "bandwidth : " << size * 1.f / cost_time << " GB/s" << std::endl;

  for(int i = 0; i < 20; ++i) std::cout << "*";
  std::cout << std::endl;


  cl_event read_event;
  checkError(clEnqueueReadBuffer(queue, a_buffer, CL_TRUE, 0, size, b_host.data(), 0, nullptr, &read_event));
  checkError(clWaitForEvents(1, &read_event));
  clFinish(queue);
  clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START,
    sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END,
    sizeof(time_end), &time_end, nullptr);
  cost_time = time_end - time_start;

  std::cout << "clEnqueueReadBuffer : " << std::endl;
  std::cout << "cost time : " << cost_time  / 1e6 << " ms "<< std::endl;
  std::cout << "bandwidth : " << size * 1.f / cost_time << " GB/s" << std::endl;

  for(int i = 0; i < 20; ++i) std::cout << "*";
  std::cout << std::endl;

  cl_event map_event;
  char* ptr = (char*) clEnqueueMapBuffer(queue, a_buffer, CL_TRUE, CL_MAP_READ, 0, size, 0, nullptr, &map_event, &err);
  checkError(err);
  checkError(clWaitForEvents(1, &map_event));
  clFinish(queue);
  clGetEventProfilingInfo(map_event, CL_PROFILING_COMMAND_START,
    sizeof(time_start), &time_start, nullptr);
  clGetEventProfilingInfo(map_event, CL_PROFILING_COMMAND_END,
    sizeof(time_end), &time_end, nullptr);
  cost_time = time_end - time_start;
  std::cout << "clEnqueueMapBuffer : " << std::endl;
  std::cout << "cost time : " << cost_time  / 1e6 << " ms "<< std::endl;

  checkError(clReleaseMemObject(a_buffer));
  return 0;
}
