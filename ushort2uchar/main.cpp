#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "arm_neon.h"

#define ushort unsigned short
#define uchar unsigned char
#define MAX(a, b) (a) > (b) ? (a) : (b)
#define CLAMP(a) (a) < 0 ? 0 : ( a > 255 ? 255 : a)

void randomRangeData(ushort* data, int len, int maxNum = 255) {
    size_t tmp;
    clock_t ct = clock();
    srand((unsigned int)ct);

    for(int i = 0; i < len; ++i) {
        tmp = rand() % maxNum;
        data[i] = (ushort) tmp;
    }
}

void common_short2uchar(const ushort* in, uchar* out, int len) {
    if(len <= 0) {
        return;
    }

    ushort max_value = 0;
    for(int i = 0; i < len; ++i) {
        max_value = MAX(max_value, in[i]);
    }

    if(0 == max_value) {
        return;
    }

    const float coeff = 255 / sqrt(max_value);
    for(int i = 0; i < len; ++i) {
        float tmp = sqrt(in[i]) * coeff;
        uchar result = tmp;
        out[i] = result;
    }
}

void common_short2uchar_neon(const ushort* in, uchar* out, int len) {
    if(len <= 0) {
        return;
    }

    ushort max_value = 0;
    for(int i = 0; i < len; ++i) {
        max_value = MAX(max_value, in[i]);
    }

    if(0 == max_value) {
        return;
    }

    const float coeff = 255 / sqrt(max_value);
    float32x4_t coeff_;
    coeff_ = vdupq_n_f32(coeff);
    uint16x8_t v_zeros = vdupq_n_u16(0);
    uint16x8_t v_255 = vdupq_n_u16(255);

    int i = 0;
    for(; i < len - 8; i += 8) {
        uint16x8_t data = vld1q_u16(in + i);
        float32x4_t data_high, data_low;
        data_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(data)));
        data_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(data)));
        float32x4_t res_high = vrecpeq_f32(vrsqrteq_f32(data_high));
        float32x4_t res_low  = vrecpeq_f32(vrsqrteq_f32(data_low));
        res_high = vmulq_f32(res_high, coeff_);
        res_low  = vmulq_f32(res_low, coeff_);

        uint16x4_t tmp_high = vmovn_u32(vcvtq_u32_f32(res_high));
        uint16x4_t tmp_low  = vmovn_u32(vcvtq_u32_f32(res_low));
        //uint16x4_t tmp_high = vmovn_u32(vcvtq_u32_f32(vaddq_f32(res_high, vdupq_n_f32(0.5f))));
        //uint16x4_t tmp_low  = vmovn_u32(vcvtq_u32_f32(vaddq_f32(res_low, vdupq_n_f32(0.5f))));
        //uint16x4_t tmp_high = vmovn_u32(vcvtq_u32_f32(vsubq_f32(res_high, vdupq_n_f32(0.5f))));
        //uint16x4_t tmp_low  = vmovn_u32(vcvtq_u32_f32(vsubq_f32(res_low, vdupq_n_f32(0.5f))));

        uint16x8_t val = vcombine_u16(tmp_low, tmp_high);
        val = vmaxq_u16(v_zeros, vminq_u16(val, v_255));
        uint8x8_t result = vqmovn_u16(val);
        vst1_u8(out + i, result);
    }
    for(; i < len; ++i) {
        float tmp = sqrt(in[i]) * coeff;
        uchar result = CLAMP(tmp);
        out[i] = result;
    }
}

double uniGetTime() {
    struct timespec cl_time;
    clock_gettime(CLOCK_MONOTONIC, &cl_time);
    double time = (double)cl_time.tv_sec + cl_time.tv_nsec * 1e-9;
    return time;
}

int main() {
    const int len = 640 * 480;
    ushort in[len];
    uchar out1[len];
    uchar out2[len];
    randomRangeData(in, len);

    double start = uniGetTime();
    common_short2uchar(in, out1, len);
    double end = uniGetTime();
    double duration = end - start;
    printf("orign time %f\n", duration);

    start = uniGetTime();
    common_short2uchar_neon(in, out2, len);
    end = uniGetTime();
    duration = end - start;
    printf("neon  time %f\n", duration);

    bool flag = true;
    for(int i = 0; i < len; ++i) {
        if(abs(out1[i] - out2[i]) > 1) {
            flag = false;
            break;
        }
        //printf("out %d : %d %d diff %d\n", i, out1[i], out2[i], out1[i] - out2[i]);
    }
    if(flag) {
        printf("result is right\n");
    }
    return 0;
}
