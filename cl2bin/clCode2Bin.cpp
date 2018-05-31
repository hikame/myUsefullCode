#include <fstream>
#include <iostream>
#include "CL/cl.h"
#include "cmdline/cmdline.h"
using namespace std;
#define uchar unsigned char

#define checkError(err) \
    if(CL_SUCCESS != err) { \
        cout << "line " << __LINE__ << " : error code " << err << endl; \
    }

string readFileIntoString(string input_name) {
    ifstream ifile(input_name.c_str());
    ostringstream buf;
    char ch;
    while(buf && ifile.get(ch)) {
        buf.put(ch);
    }
    return buf.str();
}

int main(int argc, char** argv) {
    cmdline::parser ps;
    ps.add<string>("input", 'i', "input name", true, "");
    ps.add<string>("output", 'o', "output name", true, "");
    ps.add<string>("options", 'O', "options", false, "");
    ps.add<int>("device", 'd', "device", false, 0);

    ps.parse_check(argc, argv);

    string input_name = ps.get<string>("input");
    string output_name = ps.get<string>("output");
    string options_name = ps.get<string>("options");
    int device = ps.get<int>("device");
    cout << "device " << device << " : " << input_name << " --> " << output_name;
    if(ps.exist("options"))
        cout << " ( " << options_name << " )";
    cout << endl;

    cl_uint num_platforms;
    cl_platform_id* platforms;
    cl_uint num_devices;
    cl_device_id* devices;
    cl_context context;
    cl_command_queue queue;
    cl_int  err;
    cl_program program;
    cl_kernel kernel;

    checkError(clGetPlatformIDs(0, 0, &num_platforms));
    platforms = (cl_platform_id*) malloc(num_platforms * sizeof(cl_platform_id));
    checkError(clGetPlatformIDs(num_platforms, platforms, NULL));
    checkError(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices));
    devices = (cl_device_id*) malloc(num_devices * sizeof(cl_device_id));
    checkError(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL));
    context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
    checkError(err);

    ifstream istrm;
    istrm.open(input_name.c_str(), ios::in);
    if(!istrm.is_open()) {
        cout << "Failed to open " << input_name << endl;
        return -1;
    }
    string input_str = readFileIntoString(input_name);
    const char* src = input_str.c_str();

    program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    checkError(err);
    checkError(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

    size_t* programBinarySizes = (size_t*) malloc(num_devices * sizeof(size_t));
    checkError(clGetProgramInfo(program,
                     CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t) * num_devices,
                     programBinarySizes,
                     NULL));

    uchar** programBinaries = (uchar**) malloc(num_devices * sizeof(uchar*));
    for(int i = 0; i < num_devices; ++i) {
        programBinaries[i] = (uchar*) malloc(programBinarySizes[i]);
    }
    checkError(clGetProgramInfo(program,
                    CL_PROGRAM_BINARIES,
                    sizeof(uchar*) * num_devices,
                    programBinaries,
                    NULL));

    ofstream ostm(output_name.c_str());
    ostm.write((const char*)programBinaries[device], programBinarySizes[0]);

    free(platforms);
    free(devices);
    for(int i = 0; i < num_devices; ++i) {
        free(programBinaries[i]);
    }
    free(programBinarySizes);
    free(programBinaries);
    checkError(clReleaseProgram(program))
    checkError(clReleaseContext(context))
    return 0;
}
