#ifndef HPC_UNIVERSAL_CONFIG_H
#define HPC_UNIVERSAL_CONFIG_H

// Platform based configuration
// we use HPC_OS to specify which platform we use

//we don't support 32 bit windows and unix
#define HPC_LINUX       0x00
#define HPC_WIN         0x01
#define HPC_ANDROID     0x02
#define HPC_IOS         0x03
#define HPC_MAC         0x04

// we use HPC_COMPILER to specify which compiler we use
#define HPC_MSVC        0x10
#define HPC_GCC         0x20
#define HPC_CLANG       0x30
#define HPC_ICC         0x40

#if defined(_WIN64) ||defined(_WIN32)
#define HPC_OS HPC_WIN
#endif

#if defined(__linux__)
#if defined(__ANDROID__)
#define HPC_OS HPC_ANDROID
#else
#define HPC_OS HPC_LINUX
#endif
#endif

#if defined(__APPLE__)
#define HPC_OS HPC_MAC
#endif

#if defined(__IOS__)
#define HPC_OS HPC_IOS
#endif

#if defined(_MSC_VER)
#if _MSC_VER < 1600
#error Microsoft Visual C++ of version lower than MSVC 2010 is not supported.
#endif

#define HPC_COMPILER HPC_MSVC

#elif (defined(__GNUC__))

#if (defined(__clang__))
#if ((__clang_major__ < 3))
#error CLANG of version lower than 3.0 is not supported
#endif
#define HPC_COMPILER HPC_CLANG

#else
#if ((__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 6))
#error GCC of version lower than 4.6.0 is not supported
#endif
#if (defined(__INTEL_COMPILER))
#define HPC_COMPILER HPC_ICC
#else
#define HPC_COMPILER HPC_GCC
#endif
#endif

#else
#error We only be used with Microsoft Visual C++, GCC (G++), or clang (clang++).
#endif




/**
 * Default alignment specification (library-wise).
 */
#define HPC_ALIGNMENT 64

#endif
