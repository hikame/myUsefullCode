#ifndef HPC_UNIVERSAL_FILE_H_
#define HPC_UNIVERSAL_FILE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @file
 */
/**
 * @brief load all contents in bin file and return them
 *
 * @param filename the name of bin file you want to load
 * @param s file content return, please free it in your code
 * @param length length of content if != NULL
 *
 * @return
 * 	-1 error
 * 	0 success
 *
 * @warning please free the memory allocated in this function
 *
 */
inline HPCStatus_t uniLoadBinFile(const char *filename, char** s, size_t* length) {
    if((NULL != *s) || (NULL == filename) || (0 == strlen(filename)) ) return HPC_INVALID_ARGS;

    FILE *fp = fopen(filename, "rb");
    if(NULL == fp) return HPC_IOERROR;

    rewind(fp);
    if(0 != fseek(fp, 0, SEEK_END)) return HPC_IOERROR;

    int len = ftell(fp);
    if(NULL != length) *length = len;

    char* str = (char*) malloc(len);
    if(NULL == str) return HPC_ALLOC_FAILED;

    rewind(fp);

    if(len != fread(str, sizeof(char), len, fp)){
        free(str);
        fclose(fp);
        return HPC_IOERROR;
    }

    *s = str;

    fclose(fp);

    return HPC_SUCCESS;
}

inline HPCStatus_t uniStoreToBinFile(const char *filename, size_t length, const unsigned char* s) {
    if((NULL == s) || (NULL == filename) || (0 == strlen(filename)) ) return HPC_INVALID_ARGS;

    FILE *fp = fopen(filename, "wb");
    if(NULL == fp) return HPC_IOERROR;

    if(length != fwrite(s, sizeof(char), length, fp)) {
        fclose(fp);
        return HPC_IOERROR;
    }

    fclose(fp);

    return HPC_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif
