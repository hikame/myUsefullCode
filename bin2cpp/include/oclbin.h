#ifndef _BIN2CPP_H
#define _BIN2CPP_H

#include <map>
#include <string>
using namespace std;
#define uchar unsigned char

typedef struct {
        const uchar bin;
        const size_t size;
    }clKernel;

extern const map<string, clKernel> allOCLKernels;

#endif
