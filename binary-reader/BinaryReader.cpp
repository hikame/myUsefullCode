#include <iostream>
#include <fstream>
#include <cstring>
#include <initializer_list>
#include <cmath>
#include <vector>


class BinaryReader {
public:
    BinaryReader(std::string fileName) {
        this->fileName = fileName;
    }

    int getFileSize(const char* fileName) {
        FILE *fp = fopen(fileName, "r");
        if (!fp) return -1;
        fseek(fp, 0L, SEEK_END);
        int size = ftell(fp);
        fclose(fp);
        return size;
    }

    int readFile(u_char** ptr) {
        int fileSize = getFileSize(fileName.c_str());
        if(fileSize < 0) {
            return -1;
        }

        bufferPtr = new u_char[fileSize];
        if(bufferPtr == nullptr) {
            return -1;
        }

        std::ifstream reader(fileName);
        if(reader.is_open()) {
            reader.read((char *)bufferPtr, fileSize);
            *ptr = bufferPtr;
            return 0;
        } else {
            std::cout << "The File" << fileName << " is not found." << std::endl;
            return -1;
        }
    }

    ~BinaryReader() {
        if(bufferPtr != nullptr) {
            delete [] bufferPtr;
            bufferPtr = nullptr;
        }
    }
private:
    std::string fileName = "";
    int fileSize = 0;
    u_char * bufferPtr = nullptr;
};

int main () {
    std::cout << "Test Conv2d Quantization Begin" << std::endl;

    u_char * inputPtr;
    BinaryReader inputReader("input.1.bin");
    inputReader.readFile(&inputPtr);

    return 0;
}
