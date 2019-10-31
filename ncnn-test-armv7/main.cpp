#include "ncnn/net.h"
#include "ncnn/cpu.h"
#include "ncnn/opencv.h"
#include "cmdline.h"


#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
int main(int argc, char** argv) {

  cmdline::parser psr;
  psr.add<string>("input", 'i', "input.bin [must have]", true, "");
  psr.add<string>("param", 'p', "xxx.ncnn.param [must have.]", true, "");
  psr.add<string>("iblob", 'x', "input blob name [must have.]", true, "");
  psr.add<string>("oblob", 'y', "output blob name [must have.]", true, "");
  psr.add<string>("model", 'm', "xxx.ncnn.bin", false, "");
  psr.add<int>("height", 'h', "input height[must have]", true, 256);
  psr.add<int>("width", 'w', "input width[must have]", true, 256);
  psr.add<int>("loop", 'l', "loop num", false, 30);
  psr.add<int>("thread", 't', "thread num", false, 1);
  psr.add("save", '\0', "save ncnn result");

  psr.parse_check(argc, argv);

  cout << "src          : " << psr.get<string>("input") << endl;
  cout << "param        : " << psr.get<string>("param") << endl;
  cout << "input blob   : " << psr.get<string>("iblob") << endl;
  cout << "output blob  : " << psr.get<string>("oblob") << endl;
  cout << "thread       : " << psr.get<int>("thread") << endl;
  cout << "loop         : " << psr.get<int>("loop") << endl;
  if(psr.exist("model")) {
    cout << "model        : " << psr.get<string>("model") << endl;
  }
  cout  << endl;

  string imagePath = psr.get<string>("input");
  ifstream reader;
  reader.open(imagePath, ifstream::binary);
  if(!reader.is_open()) {
    cout << imagePath << " not exist" << endl;
    return -1;
  }

  cout << "reading data" << endl;
  vector<float> buffer;
  float tmp = 0;
  while(reader.read((char*)&tmp, sizeof(float))) {
    buffer.push_back(tmp);
  }
  reader.close();

  int num = buffer.size();
  cout << "input size (byte) : " << buffer.size() << endl;

  int row = psr.get<int>("height");
  int col = psr.get<int>("width");
  int channel = 3;
  ncnn::Mat src = ncnn::Mat(row, col, channel, buffer.data());


  ncnn::Net ncnnNet;
  ncnn::Mat dst;
  ncnnNet.load_param(psr.get<string>("param").c_str());

  if(psr.exist("model")) {
    ncnnNet.load_model(psr.get<string>("model").c_str());
  }


  std::string iblob = psr.get<string>("iblob");
  std::string oblob = psr.get<string>("oblob");

  // run model
  {
    ncnn::Extractor ex= ncnnNet.create_extractor();
    ex.input(iblob.c_str(), src);
    ex.extract(oblob.c_str(), dst);
  }


  if(psr.exist("save")) {
    ofstream writer;
    writer.open("./ncnn.output.dat", ofstream::binary);
    writer.write(reinterpret_cast<char*>(dst.data), sizeof(float) * num);
    writer.close();
  }

  return 0;
}
