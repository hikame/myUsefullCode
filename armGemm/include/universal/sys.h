#ifndef HPC_UNIVERSAL_SYS_H_
#define HPC_UNIVERSAL_SYS_H_

#include <stdlib.h>
#include "status.h"

#if defined (USE_OCL)
#include "CL/cl.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Programming ECO System
 **/
typedef enum {
    ECO_ENV_X86  = 1,
    ECO_ENV_ARM  = 2,
    ECO_ENV_CUDA = 4,
    ECO_ENV_OCL  = 8,
} EcoEnv_t;

/**
 * @biref System Platform
 */
typedef enum {
    PLATFORM_TYPE_INTEL     = 0,
    PLATFORM_TYPE_AMD       = 1,
    PLATFORM_TYPE_NVIDIA    = 2,
    PLATFORM_TYPE_ARM       = 3,
    PLATFORM_TYPE_QUALCOMM  = 4,
} PlatformType_t;

/**
 * @brief Instruction Set Architecture (ISA)
 *
 */
typedef enum {
    ISA_NOT_SUPPORTED       = 0x0,
#if defined(USE_X86)
    ISA_X86_SSE             = 0x1,
    ISA_X86_SSE2            = 0x2,
    ISA_X86_SSE3            = 0x4,
    ISA_X86_SSSE3           = 0x8,
    ISA_X86_SSE41           = 0x10,
    ISA_X86_SSE42           = 0x20,
    ISA_X86_AVX             = 0x40,
    ISA_X86_AVX2            = 0x80,
    ISA_X86_FMA             = 0x100,
#endif
#if defined(USE_ARM)
    ISA_ARM_V7              = 0x1,
    ISA_ARM_V8              = 0x2,
#endif
#if defined(USE_CUDA)
    ISA_NV_KEPLER           = 0x1,
    ISA_NV_MAXWELL          = 0x2,
#endif
#if defined(USE_OCL)
    ISA_AMD_GCN             = 0x1,
    ISA_ARM_MALI            = 0x2,
#endif
} ISA_t;

/**
 * @brief Implementation (IMP)
 */
typedef enum {
    IMP_NOT_SUPPORTED    = -1,
#if defined(USE_X86)
    IMP_X86_SANDYBRIDGE  = 0,
    IMP_X86_IVYBRIDGE    = 1,
    IMP_X86_HASWELL      = 12,
    IMP_X86_BROADWELL    = 13,
    IMP_X86_SKYLAKE      = 23,
#endif
#if defined(USE_ARM)
    IMP_ARM_A9           = 110,
    IMP_ARM_A15          = 111,
    IMP_ARM_A53          = 120,
    IMP_ARM_A57          = 121,
#endif
#if defined(USE_CUDA)
    IMP_NV_SM30          = 210,
    IMP_NV_SM32          = 211,
    IMP_NV_SM35          = 212,
    IMP_NV_SM50          = 220,
    IMP_NV_SM52          = 221,
#endif
#if defined(USE_OCL)
    IMP_AMD_GCN10        = 310,
    IMP_AMD_GCN11        = 311,
    IMP_AMD_GCN12        = 312,
#endif
} IMP_t;

/**
 * @brief Properties
 */
#if defined(USE_X86)
typedef struct {
    // char name[256];
    int core;
    unsigned long isa;
    //IMP_t imp;

    /*int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t l1CacheSize;
    size_t l1CacheLineSize;
    size_t l2CacheSize;
    size_t l2CacheLineSize;
    size_t l3CacheSize;
    size_t l3CacheLineSize;*/
} X86Properties;

float uniTestIsaFMA();
float uniTestIsaAVX();
float uniTestIsaSSE();
bool uniCheckRuntimeStatus(int core, ISA_t isa);

bool uniX86GetProperties(int core, X86Properties *p);

X86Properties* uniX86GetPlatInfo(int core);

/**
 * @brief which isa windows support
 */
bool uniX86SetISA_t(int isa);

#endif

//comment this because we can't find a way to get all of them
#if defined(USE_ARM)
typedef struct {
    // char name[256];
    int core;
    ISA_t isa;
    //IMP_t imp;

    /*int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t l1CacheSize;
    size_t l1CacheLineSize;
    size_t l2CacheSize;
    size_t l2CacheLineSize;*/
} ARMProperties;

bool uniARMGetProperties(int core, ARMProperties *p);

#endif

#if defined(USE_CUDA)
typedef struct {
    char name[256];
    ISA_t isa;
    IMP_t imp;

    int cc; //compute capability, 4 digits, 3050 for 3.5;5020 for 5.2
    int numCores; //num of compute cores
    int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t sharedSizeInBytes;
    size_t totalDramSize;
} CUDAProperties;

bool uniCUDAGetProperties(int device, CUDAProperties *p);

bool uniCUDAGetFlops(const CUDAProperties *p, float *baseGflops, float* maxGflops);
#endif

#if defined(USE_OCL)
typedef struct {
    char name[256];
    ISA_t isa;
    IMP_t imp;

    int cc; //compute capability, 4 digits, 3050 for 3.5;5020 for 5.2
    int numCores; //num of compute cores
    int numFlopsPerClockPerCore;
    float baseFrequency;// GHz
    float maxFrequency;// GHz

    size_t sharedSizeInBytes;
    size_t totalDramSize;
} OCLProperties;

bool uniOCLGetProperties(int device, OCLProperties *p);

// bool uniOCLGetFlops(const OCLProperties *p, float *baseGflops, float* maxGflops);
#endif

/**
 * @brief bind current thread to one processor
 *
 * @param coreId          processor ID for current thread to bind
 */
#if defined(USE_X86)
bool uniX86BindThreadToCore(int coreId);
#endif

#if defined(USE_ARM)
bool uniARMBindThreadToCore(int coreId);
#endif

/**
 * @brief Malloc and free data
 */
#if defined(USE_X86)
bool uniX86Malloc(void** d, size_t size);

bool uniX86Free(void* d);
#endif


#if defined(USE_ARM)
bool uniARMMalloc(void** d, size_t size);

bool uniARMFree(void* d);
#endif

#if defined(USE_CUDA)
bool uniCUDAMalloc(void** d, size_t size);

bool uniCUDAFree(void* d);
#endif

#if defined(USE_OCL)
bool uniOCLMalloc(cl_context context, cl_mem_flags flags, cl_mem *d, size_t size);

bool uniOCLFree(cl_mem d);
#endif

#if defined(USE_CUDA)
HPCStatus_t uniHostCopy1dToDevice(int device, cudaStream_t stream, size_t numBytes, const void* src, void* dst);

HPCStatus_t uniHostCopy1dFromDevice(int device, cudaStream_t stream, size_t numBytes, const void* src, void* dst);

HPCStatus_t uniHostCopy2dToDevice(int device, cudaStream_t stream, size_t height, size_t widthInBytes, size_t widthOfSrcInBytes, const void* src, size_t widthOfDstInBytes, void* dst);

HPCStatus_t uniHostCopy2dFromDevice(int device, cudaStream_t stream, size_t height, size_t widthInBytes, size_t widthOfSrcInBytes, const void* src, size_t widthOfDstInBytes, void* dst);

HPCStatus_t uniDeviceCopy1dToDevice(int device, cudaStream_t stream, size_t numBytes, const void* src, void* dst);

HPCStatus_t uniDeviceSetMemory(int device, cudaStream_t stream, size_t numBytes, int value, void* mem);

HPCStatus_t uniDeviceCopy1dPeer(cudaStream_t stream, size_t numBytes, int srcDevice, const void* src, int dstDevice, void* dst);

HPCStatus_t uniDeviceCopy2dToDevice(int device, cudaStream_t stream, size_t height, size_t widthInBytes, size_t widthOfSrcInBytes, const void* src, size_t widthOfDstInBytes, void* dst);

HPCStatus_t uniCUDASynchronizeStream(cudaStream_t stream);

HPCStatus_t uniCUDASynchronizeDevice(int device);

#endif

#ifdef __cplusplus
}
#endif

#endif

