#include "cmdline.h"
#include "duration.hpp"
#include "ncnn/cpu.h"
#include "ncnn/net.h"
#include "ncnn/opencv.h"

#include <fstream>
#include <iostream>
#include <pthread.h>
#include <vector>

using namespace std;

int main(int argc, char **argv) {

  cmdline::parser psr;
  psr.add<string>("input", 'i', "input.bin [must have]", true, "");
  psr.add<string>("param", 'p', "xxx.ncnn.param [must have.]", true, "");
  psr.add<string>("iblob", 'x', "input blob name [must have.]", true, "");
  psr.add<string>("oblob", 'y', "output blob name [must have.]", true, "");
  psr.add<string>("model", 'm', "xxx.ncnn.bin", true, "");
  psr.add<int>("height", 'h', "input height[must have]", true, 256);
  psr.add<int>("width", 'w', "input width[must have]", true, 256);
  psr.add<int>("loop", 'l', "loop num", false, 30);
  psr.add<int>("thread", 't', "thread num", false, 1);
  psr.add<int>("core", 'c', "core id", false, 0);
  psr.add("save", '\0', "save ncnn result");

  psr.parse_check(argc, argv);

  cout << "src          : " << psr.get<string>("input") << endl;
  cout << "param        : " << psr.get<string>("param") << endl;
  cout << "input blob   : " << psr.get<string>("iblob") << endl;
  cout << "output blob  : " << psr.get<string>("oblob") << endl;
  cout << "thread       : " << psr.get<int>("thread") << endl;
  cout << "loop         : " << psr.get<int>("loop") << endl;
  if (psr.exist("model")) {
    cout << "model        : " << psr.get<string>("model") << endl;
  }
  cout << endl;

  int num_thread = psr.get<int>("thread");
  if (psr.exist("core")) {
    int core = psr.get<int>("core");
    cout << "core         : " << core << endl;
    // bind core to do.
  }

  string imagePath = psr.get<string>("input");
  ifstream reader;
  reader.open(imagePath, ifstream::binary);
  if (!reader.is_open()) {
    cout << imagePath << " not exist" << endl;
    return -1;
  }

  cout << "reading data" << endl;
  vector<float> buffer;
  float tmp = 0;
  while (reader.read((char *)&tmp, sizeof(float))) {
    buffer.push_back(tmp);
  }
  reader.close();

  int num = buffer.size();
  cout << "input size    : " << buffer.size() * sizeof(float) << " (byte)"
       << endl;

  int row = psr.get<int>("height");
  int col = psr.get<int>("width");
  int channel = 3;
  ncnn::Mat src = ncnn::Mat(row, col, channel, buffer.data());
  cout << "input height  : " << row << endl;
  cout << "input width   : " << row << endl;
  cout << "input channel : " << channel << endl;
  if (row * col * channel != num) {
    cout << "height * width * channel != " << imagePath << endl;
    exit(0);
  }

  ncnn::Net ncnnNet;
  ncnn::Mat dst;
  ncnnNet.load_param(psr.get<string>("param").c_str());
  ncnnNet.load_model(psr.get<string>("model").c_str());

  std::string iblob = psr.get<string>("iblob");
  std::string oblob = psr.get<string>("oblob");

  // warm up.
  // run model
  for (int i = 0; i < 10; ++i) {
    ncnn::Extractor ex = ncnnNet.create_extractor();
    ex.set_num_threads(num_thread);
    ex.input(iblob.c_str(), src);
    ex.extract(oblob.c_str(), dst);
  }

  int loop = psr.get<int>("loop");

  Duration tt;
  tt.start();
  for (int i = 0; i < loop; ++i) {
    ncnn::Extractor ex = ncnnNet.create_extractor();
    ex.set_num_threads(num_thread);
    ex.input(iblob.c_str(), src);
    ex.extract(oblob.c_str(), dst);
  }
  tt.end();

  float ave_time = tt.getDuration() / loop;
  cout << "ave time : " << ave_time << endl;

  if (psr.exist("save")) {
    ofstream writer;
    writer.open("./ncnn.output.dat", ofstream::binary);
    writer.write(reinterpret_cast<char *>(dst.data), sizeof(float) * num);
    writer.close();
  }

  return 0;
}
