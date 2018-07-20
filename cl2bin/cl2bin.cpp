#include <fstream>
#include <iostream>
#include "CL/cl.h"
#include "cmdline.h"
using namespace std;
#define uchar unsigned char

#define checkError(err) \
  if(CL_SUCCESS != err) { \
    cout << "line " << __LINE__ << " : error code " << err << endl; \
  }

void deleteAllMark(string & s, const string & mark) {
  size_t nSize = mark.size();
  while(1) {
    size_t pos = s.find(mark);
    if(string::npos == pos) {
      return;
    }
    s.erase(pos, nSize);
  }
}

vector< string > getOptionValue(string & s, const string & mark) {
  vector< string > res;
  while(1) {
    size_t pos = s.find(mark);
    if(string::npos == pos) {
      deleteAllMark(s, " ");
      res.push_back(s);
      return res;
    }
    string tmp = s.substr(0, pos);
    deleteAllMark(tmp, " ");
    res.push_back(tmp);
    s = s.substr(pos + 1);
  }
}

void parseOptions(string & s, vector< vector<string> > & opt_value,
              vector<string>& opt_name) {
  while(1) {
    int pos = s.find("OPTION");
    if(string::npos == pos) {
      return;
    }
    int name_pos = s.find("=", pos);
    int blank_pos = s.find(" ", pos);
    string option_name = s.substr(blank_pos, name_pos - blank_pos);
    deleteAllMark(option_name, " ");
    opt_name.push_back(option_name);


    int left_idx = s.find("{");
    int right_idx = s.find("}");
    string val_str = s.substr(left_idx + 1, right_idx - left_idx - 1 );
    vector<string> res = getOptionValue(val_str, ",");
    opt_value.push_back(res);
    s = s.substr(right_idx + 1);
  }
}

void combination(vector< vector<string> >& s, string pre,
                vector<string>& result, int row, int col) {
  if(row < s.size()) {
    for(; col < s[row].size(); ++col) {
      string tmp = pre + s[row][col];
      combination(s, tmp, result, row + 1, 0);
    }
  }
  if(row == s.size()) {
    result.push_back(pre);
  }
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
  std::cout << "num_devices " << num_devices << std::endl;
  context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
  checkError(err);

  ifstream istrm;
  istrm.open(input_name.c_str(), ios::in);
  if(!istrm.is_open()) {
    cout << "Failed to open " << input_name << endl;
    return -1;
  }
  string input_str = readFileIntoString(input_name);

  //parse options in the xxx.cl files
  vector< vector<string> > opt_values;
  vector<string> opt_names;
  vector<string> opt_cmbs;
  vector<string> opt_res;
  parseOptions(input_str, opt_values, opt_names);
  combination(opt_values, "", opt_cmbs, 0, 0);
  for(int i = 0; i < opt_cmbs.size(); ++i) {
    string tmp = "";
    for(int j = 0; j < opt_names.size(); ++j) {
      tmp += "-D" + opt_names[j] + "=" + opt_cmbs[i].c_str()[j] + " ";
    }
    opt_res.push_back(tmp);
  }

  const char* src = input_str.c_str();

  program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
  checkError(err);

  size_t* programBinarySizes = (size_t*) malloc(num_devices * sizeof(size_t));
  uchar** programBinaries = (uchar**) malloc(num_devices * sizeof(uchar*));

  cout << "opt_res size " << opt_res.size() << std::endl;
  for(int n = 0; n < opt_res.size(); ++n) {
    string cl_opt = options_name + opt_res[n];
    cout << cl_opt << endl;
    err = clBuildProgram(program, num_devices, devices, cl_opt.c_str(), NULL, NULL);
    if(CL_SUCCESS != err) {
      cout << "clBuildProgram err" << endl;
      size_t log_size;
      checkError( clGetProgramBuildInfo(program, devices[device], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
      char* log = (char*) malloc(log_size + 1);
      log[log_size] = '\0';
      checkError( clGetProgramBuildInfo(program, devices[device], CL_PROGRAM_BUILD_LOG, log_size + 1, log, NULL) );
      cout << "err : " << endl;
      cout << log;
      free(log);
      return -1;
    }

    checkError(clGetProgramInfo(program,
          CL_PROGRAM_BINARY_SIZES,
          sizeof(size_t) * num_devices,
          programBinarySizes,
          NULL));

    for(int i = 0; i < num_devices; ++i) {
      programBinaries[i] = (uchar*) malloc(programBinarySizes[i]);
    }
    checkError(clGetProgramInfo(program,
          CL_PROGRAM_BINARIES,
          sizeof(uchar*) * num_devices,
          programBinaries,
          NULL));

    string suffix_name;
    if(opt_res.size() == 1)
      suffix_name = opt_cmbs[n] + ".bin";
    else {
      suffix_name = "_" + opt_cmbs[n] + ".bin";
    }
    string tmp_name = output_name + suffix_name;
    ofstream ostm(tmp_name.c_str());
    ostm.write((const char*)programBinaries[device], programBinarySizes[0]);
    for(int i = 0; i < num_devices; ++i) {
      free(programBinaries[i]);
    }
  }

  free(programBinaries);
  free(platforms);
  free(devices);
  free(programBinarySizes);
  checkError(clReleaseProgram(program));
  checkError(clReleaseContext(context));
  return 0;
}
