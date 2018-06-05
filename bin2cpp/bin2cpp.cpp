#include <fstream>
#include <string>
#include <iostream>
#include "cmdline.h"

#define uchar unsigned char
using namespace std;


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
    ps.parse_check(argc, argv);
    string input_name = ps.get<string>("input");
    string output_name = ps.get<string>("output");

    //cout << "input : " << input_name << " output : " << output_name << endl;
    string input_str = readFileIntoString(input_name);
    int last_dot = input_name.rfind(".");
    int next_backslash = input_name.rfind("/") + 1;
    string file_name = input_name.substr(next_backslash, last_dot - next_backslash);
    //cout << "file_name : " << file_name << endl;
    stringstream len_str;
    len_str << input_str.size();
    string output_str = "#include \"oclbin.h\"\n";
    output_str += "const size_t " + file_name + "_len = " + len_str.str() + ";\n";
    output_str += "const uchar " + file_name + "[] = {\n";

    ofstream ostrm;
    ostrm.open(output_name.c_str());
    if(!ostrm.is_open()) {
        cout << "Failed to open " << output_name << endl;
        return -1;
    }
    uchar aChar;
    string appendStr;
    for(int i = 0; i < input_str.size(); ++i) {
         appendStr.clear();
         if(0 == (i % 15) && 0 != i) {
            appendStr += "\n";
         }
         char tmp[4];
         aChar = input_str.c_str()[i];
         sprintf(tmp, "0x%02x", aChar);
         appendStr += string(tmp);
         if(i != input_str.size() - 1) {
            appendStr += ",";
         }
        output_str += appendStr;
    }
    output_str += "};";

    ostrm.write((const char*)output_str.c_str(), output_str.size());

    ostrm.close();
    return 0;
}
