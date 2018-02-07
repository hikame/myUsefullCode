#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "arm_neon.h"

#define USE_ARM
#include "fmath/matrix.hpp"
using namespace HPC::fmath;

#define mm_req_blocking \
    v_a_high = vget_high_f32(v_a);                      \
    v_a_low  = vget_low_f32(v_a);                       \
    v_c0 = vmlaq_lane_f32(v_c0, v_b0, v_a_low,  0);     \
    v_c1 = vmlaq_lane_f32(v_c1, v_b0, v_a_low,  1);     \
    v_c2 = vmlaq_lane_f32(v_c2, v_b0, v_a_high, 0);     \
    v_c3 = vmlaq_lane_f32(v_c3, v_b0, v_a_high, 1);     \
    v_c4 = vmlaq_lane_f32(v_c4, v_b1, v_a_low,  0);     \
    v_c5 = vmlaq_lane_f32(v_c5, v_b1, v_a_low,  1);     \
    v_c6 = vmlaq_lane_f32(v_c6, v_b1, v_a_high, 0);     \
    v_c7 = vmlaq_lane_f32(v_c7, v_b1, v_a_high, 1);

#define loop(k)                                         \
    v_a  = vld1q_f32(src_a + (k) * MB + i);             \
    v_b0 = vld1q_f32(src_b + (k) * NB + j);             \
    v_b1 = vld1q_f32(src_b + (k) * NB + j + 4);         \
    mm_req_blocking

#define loop2(ck) {loop(ck)  loop(ck + 1)}
#define loop4(ck) {loop2(ck) loop2(ck + 2)}

void mm_11_blocking(
        const float* __restrict__ src_a,
        const float* __restrict__ src_b,
        float*       __restrict__ dst_c,
        const int                 MB,
        const int                 KB,
        const int                 NB) {

    for(int i =0; i < MB; i += 4) {
        for(int j = 0; j < NB; j += 8) {
            float32x4_t v_c0 = vld1q_f32(dst_c + i * NB + j);
            float32x4_t v_c1 = vld1q_f32(dst_c + (i + 1) * NB + j);
            float32x4_t v_c2 = vld1q_f32(dst_c + (i + 2) * NB + j);
            float32x4_t v_c3 = vld1q_f32(dst_c + (i + 3) * NB + j);
            float32x4_t v_c4 = vld1q_f32(dst_c + i * NB + j + 4);
            float32x4_t v_c5 = vld1q_f32(dst_c + (i + 1) * NB + j + 4);
            float32x4_t v_c6 = vld1q_f32(dst_c + (i + 2) * NB + j + 4);
            float32x4_t v_c7 = vld1q_f32(dst_c + (i + 3) * NB + j + 4);

            float32x2_t v_a_high, v_a_low;
            float32x4_t v_a, v_b0, v_b1;
            for(int k = 0; k < KB; k += 4) {
                loop4(k)
            }

            vst1q_f32(dst_c + i * NB + j, v_c0);
            vst1q_f32(dst_c + (i + 1) * NB + j, v_c1);
            vst1q_f32(dst_c + (i + 2) * NB + j, v_c2);
            vst1q_f32(dst_c + (i + 3) * NB + j, v_c3);
            vst1q_f32(dst_c + i * NB + j + 4, v_c4);
            vst1q_f32(dst_c + (i + 1) * NB + j + 4, v_c5);
            vst1q_f32(dst_c + (i + 2) * NB + j + 4, v_c6);
            vst1q_f32(dst_c + (i + 3) * NB + j + 4, v_c7);
        }
    }
}

void gemmNormal(
        const float* __restrict__ src_a,
        const float* __restrict__ src_b,
        float*       __restrict__ dst_c,
        const int                 MB,
        const int                 KB,
        const int                 NB) {

    for(int i = 0; i < MB; ++i) {
        for(int j = 0; j < NB; ++j) {
            float result = 0.f;
            for(int k = 0; k < KB; ++k) {
                result += src_a[i * KB + k] * src_b[k * NB + j];
            }
            dst_c[i * NB + j] = result;
        }
    }
}

void transMatrix(
        const float* __restrict__ src,
        float*  __restrict__      dst,
        const int                 rows,
        const int                 cols) {
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void randomRangeData(float* data, int len, int maxNum = 255) {
    clock_t ct = clock();
    srand((unsigned int)ct);

    for(int i = 0; i < len; ++i) {
        float tmp = rand() % maxNum;
        data[i] = (float) tmp / maxNum;
    }
}

double uniGetTime() {
    struct timespec cl_time;
    clock_gettime(CLOCK_MONOTONIC, &cl_time);
    double time = (double)cl_time.tv_sec + cl_time.tv_nsec * 1e-9;
    return time;
}

int main(int argc, char** argv) {
    if(2 != argc && 4 != argc) {
        printf("./gemm_speed M K N or ./gemm_speed M\n");
        return -1;
    }
    int M, K, N;
    if(2 == argc) {
        M = K = N = atoi(argv[1]);
    }
    else {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    printf("M %d K %d N %d\n", M, K, N);

    float* src_a = (float*) malloc(M * K * sizeof(float));
    float* src_a_trans = (float*) malloc(M * K * sizeof(float));
    float* src_b = (float*) malloc(K * N * sizeof(float));
    float* dst_c = (float*) malloc(M * N * sizeof(float));
    float* check_c = (float*) malloc(M * N * sizeof(float));
    float* check_fmath = (float*) malloc(M * N * sizeof(float));

    randomRangeData(src_a, M * K);
    randomRangeData(src_b, K * N);

    transMatrix(src_a, src_a_trans, M, K);
    for(int i = 0; i < 10; ++i) {
        mm_11_blocking(src_a_trans, src_b, dst_c, M, K, N);
    }
    memset(dst_c, 0, M * N * sizeof(float));
    double start = uniGetTime();
    mm_11_blocking(src_a_trans, src_b, dst_c, M, K, N);
    double end = uniGetTime();
    double duration = end - start;
    double count = 2.f * M * K * N / 1e9;
    printf("mm_11_blocking time %f gflops %f\n", duration, (double)count / duration);
    //gemmNormal_trans(src_a_trans, src_b, dst_c, M, K, N);

    for(int i = 0; i < 20; ++i) {
        gemmNormal(src_a, src_b, check_c, M, K, N);
    }
    start = uniGetTime();
    gemmNormal(src_a, src_b, check_c, M, K, N);
    end = uniGetTime();
    duration = end - start;
    printf("gemmNormal time %f gflops %f\n", duration, (double)count / duration);


    int tmp_size = armGemmTNGetSize<float>(0, M, N, K, M, N, N);
    float* tmp_data = (float*)malloc(tmp_size);
    for(int i = 0; i < 20; ++i) {
        armGemm_tn<float>(0, M, N, K, 1.f, M, src_a_trans, N, src_b, tmp_data, 0.f, N, check_fmath);
    }

    memset(check_fmath, 0, M * N * sizeof(float));
    start = uniGetTime();
    armGemm_tn<float>(0, M, N, K, 1.f, M, src_a_trans, N, src_b, tmp_data, 0.f, N, check_fmath);
    end = uniGetTime();
    duration = end - start;
    printf("fmath gemm_tn time %f gflops %f\n", duration, (double)count / duration);

    bool flag = true;
    for(int i = 0; i < M * N; ++i) {
        if(abs(dst_c[i] - check_fmath[i]) > 1e-3) {
            flag = false;
            break;
        }
        //printf("%d %d -> %f %f %f\n", i / N, i % N, dst_c[i], check_c[i], dst_c[i] - check_c[i]);
    }
    if(flag) {
        printf("result is right\n");
    }
    else {
        printf("result is wrong\n");
    }

    free(src_a);
    free(src_a_trans);
    free(src_b);
    free(dst_c);
    free(check_c);
    free(check_fmath);
    free(tmp_data);
    return 0;
}
