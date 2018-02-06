#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "arm_neon.h"

int main() {
    const int len = 4;
    int a[len];
    int b[len];
    for(int i = 0; i < len; ++i) {
        a[i] = i;
    }

    int32x4_t a_neon = vld1q_s32(a);
    int32x2_t tmp_val = vdup_n_s32(10);
    int32x2_t a_high = vget_high_s32(a_neon);
    int32x2_t a_low = vget_low_s32(a_neon);
    a_high = vadd_s32(tmp_val, a_high);

    int32x4_t b_neon = vcombine_s32(a_low, a_high);
    vst1q_s32(b, b_neon);

    for(int i = 0; i < len; ++i) {
        printf("i-->%d : %d\n", i, b[i]);
    }

    return 0;
}
