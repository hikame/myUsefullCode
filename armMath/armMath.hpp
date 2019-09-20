#include "math.h"

#ifdef __aarch64__

void pengcuo_exp(const float * src, float * dst, int num) {
  const float vonef = 1.f;
  const float vhalf = 0.5f;
  const float vc_exp_hi = 88.3762626647949f;
  const float vc_exp_lo = -88.3762626647949f;
  const float vc_cephes_LOG2EF = 1.44269504088896341;
  static const float cephes_exp_p[6] = {1.9875691500E-4, 1.3981999507E-3,
                                        8.3334519073E-3, 4.1665795894E-2,
                                        1.6666665459E-1, 5.0000001201E-1};
  const float vc_cephes_exp_C1 = 0.693359375;
  const float vc_cephes_exp_C2 = -2.12194440e-4;
  const int v7f = 0x7f;

  const float  vcephes_exp_p0 = cephes_exp_p[0];
  const float  vcephes_exp_p1 = cephes_exp_p[1];
  const float  vcephes_exp_p2 = cephes_exp_p[2];
  const float  vcephes_exp_p3 = cephes_exp_p[3];
  const float  vcephes_exp_p4 = cephes_exp_p[4];
  const float  vcephes_exp_p5 = cephes_exp_p[5];
  asm volatile (
      "dup  v31.4s, %w[vonef]              \n"
      "dup  v30.4s, %w[vc_exp_hi]          \n"
      "dup  v29.4s, %w[vc_exp_lo]          \n"
      "dup  v28.4s, %w[vcephes_exp_p5]     \n"
      "dup  v27.4s, %w[vcephes_exp_p4]     \n"
      "dup  v26.4s, %w[vcephes_exp_p3]     \n"
      "dup  v25.4s, %w[vcephes_exp_p2]     \n"
      "dup  v24.4s, %w[vcephes_exp_p1]     \n"
      "dup  v23.4s, %w[vc_cephes_LOG2EF]   \n"
      "dup  v22.4s, %w[vc_cephes_exp_C1]   \n"
      "dup  v21.4s, %w[vc_cephes_exp_C2]   \n"
      "dup  v20.4s, %w[v7f]                \n"
    :
    :[vonef] "r" (vonef),
     [vhalf] "r" (vhalf),
     [vc_exp_hi] "r" (vc_exp_hi),
     [vc_exp_lo] "r" (vc_exp_lo),
     [vc_cephes_LOG2EF] "r" (vc_cephes_LOG2EF),
     [vcephes_exp_p0] "r" (vcephes_exp_p0),
     [vcephes_exp_p1] "r" (vcephes_exp_p1),
     [vcephes_exp_p2] "r" (vcephes_exp_p2),
     [vcephes_exp_p3] "r" (vcephes_exp_p3),
     [vcephes_exp_p4] "r" (vcephes_exp_p4),
     [vcephes_exp_p5] "r" (vcephes_exp_p5),
     [vc_cephes_exp_C1] "r" (vc_cephes_exp_C1),
     [vc_cephes_exp_C2] "r" (vc_cephes_exp_C2),
     [v7f] "r" (v7f)

    : "v31", "v30", "v29", "v28",
      "v27", "v26", "v25", "v24",
      "v23", "v22", "v21", "v20"
  );

  int p = 0;
  for(; p < num - 7; p += 8) {
    const float * src_ptr = src + p;
    float * dst_ptr = dst + p;
    asm volatile (
        "ldr  q0, [%[src_ptr]], #16        \n"
        "ldr  q8, [%[src_ptr]]             \n"

        "dup  v1.4s, %w[vhalf]             \n"
        "dup  v9.4s, %w[vhalf]             \n"

        "fmin v0.4s, v0.4s, v30.4s         \n"
        "fmin v8.4s, v8.4s, v30.4s         \n"

        "fmax v0.4s, v0.4s, v29.4s         \n"
        "fmax v8.4s, v8.4s, v29.4s         \n"

        "fmla v1.4s, v0.4s, v23.4s         \n"
        "fmla v9.4s, v8.4s, v23.4s         \n"

        "fcvtzs v2.4s, v1.4s               \n"
        "fcvtzs v10.4s, v9.4s              \n"

        "scvtf v3.4s, v2.4s                \n"
        "scvtf v11.4s, v10.4s              \n"

        "fcmgt v4.4s, v3.4s, v1.4s         \n"
        "fcmgt v12.4s, v11.4s, v9.4s       \n"


        "and v4.16b, v4.16b, v31.16b       \n"
        "and v12.16b, v12.16b, v31.16b     \n"

        "fsub v1.4s, v3.4s, v4.4s          \n"
        "fsub v9.4s, v11.4s, v12.4s        \n"

        "fmul v3.4s, v1.4s, v22.4s         \n"
        "fmul v11.4s, v9.4s, v22.4s        \n"

        "fmul v5.4s, v1.4s, v21.4s         \n"
        "fmul v13.4s, v9.4s, v21.4s        \n"

        "dup  v6.4s, %w[vcephes_exp_p0]    \n"
        "dup  v14.4s, %w[vcephes_exp_p0]   \n"

        "fsub v0.4s, v0.4s, v3.4s          \n"
        "fsub v8.4s, v8.4s, v11.4s         \n"

        "fsub v0.4s, v0.4s, v5.4s          \n"
        "fsub v8.4s, v8.4s, v13.4s         \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fmul v14.4s, v14.4s, v8.4s        \n"

        "fmul v5.4s, v0.4s, v0.4s          \n"
        "fmul v13.4s, v8.4s, v8.4s         \n"

        "fadd v6.4s, v6.4s, v24.4s         \n"
        "fadd v14.4s, v14.4s, v24.4s       \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fmul v14.4s, v14.4s, v8.4s        \n"

        "fadd v6.4s, v6.4s, v25.4s         \n"
        "fadd v14.4s, v14.4s, v25.4s       \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fmul v14.4s, v14.4s, v8.4s        \n"

        "fadd v6.4s, v6.4s, v26.4s         \n"
        "fadd v14.4s, v14.4s, v26.4s       \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fmul v14.4s, v14.4s, v8.4s        \n"

        "fadd v6.4s, v6.4s, v27.4s         \n"
        "fadd v14.4s, v14.4s, v27.4s       \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fmul v14.4s, v14.4s, v8.4s        \n"

        "fadd v6.4s, v6.4s, v28.4s         \n"
        "fadd v14.4s, v14.4s, v28.4s       \n"

        "fmul v6.4s, v6.4s, v5.4s          \n"
        "fmul v14.4s, v14.4s, v13.4s       \n"

        "fadd v6.4s, v6.4s, v0.4s          \n"
        "fadd v14.4s, v14.4s, v8.4s        \n"

        "fadd v6.4s, v6.4s, v31.4s         \n"
        "fadd v14.4s, v14.4s, v31.4s       \n"

        "fcvtzs v7.4s, v1.4s               \n"
        "fcvtzs v15.4s, v9.4s              \n"

        "add v7.4s, v7.4s, v20.4s          \n"
        "add v15.4s, v15.4s, v20.4s        \n"

        "shl v7.4s, v7.4s, #23             \n"
        "shl v15.4s, v15.4s, #23           \n"

        "fmul v6.4s, v6.4s, v7.4s          \n"
        "fmul v14.4s, v14.4s, v15.4s       \n"

        "str  q6, [%[dst_ptr]], #16        \n"
        "str  q14, [%[dst_ptr]]            \n"
      : [src_ptr] "+r" (src_ptr),
        [dst_ptr] "+r" (dst_ptr)
      : [vhalf] "r" (vhalf),
        [vcephes_exp_p0] "r" (vcephes_exp_p0)
      : "v0", "v1", "v2", "v3",
        "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11",
        "v12", "v13", "v14", "v15",
        "memory"
    );
  }

  for(; p < num - 3; p += 4) {
    const float * src_ptr = src + p;
    float * dst_ptr = dst + p;
    asm volatile (
        "ldr  q0, [%[src_ptr]]             \n"
        "dup  v1.4s, %w[vhalf]             \n"
        "fmin v0.4s, v0.4s, v30.4s         \n"
        "fmax v0.4s, v0.4s, v29.4s         \n"

        "fmla v1.4s, v0.4s, v23.4s         \n"
        "fcvtzs v2.4s, v1.4s               \n"
        "scvtf v3.4s, v2.4s                \n"

        "fcmgt v4.4s, v3.4s, v1.4s         \n"
        "and v4.16b, v4.16b, v31.16b       \n"

        "fsub v1.4s, v3.4s, v4.4s          \n"
        "fmul v3.4s, v1.4s, v22.4s         \n"
        "fmul v5.4s, v1.4s, v21.4s         \n"

        "dup  v6.4s, %w[vcephes_exp_p0]    \n"
        "fsub v0.4s, v0.4s, v3.4s          \n"
        "fsub v0.4s, v0.4s, v5.4s          \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fmul v5.4s, v0.4s, v0.4s          \n"
        "fadd v6.4s, v6.4s, v24.4s         \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fadd v6.4s, v6.4s, v25.4s         \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fadd v6.4s, v6.4s, v26.4s         \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fadd v6.4s, v6.4s, v27.4s         \n"

        "fmul v6.4s, v6.4s, v0.4s          \n"
        "fadd v6.4s, v6.4s, v28.4s         \n"

        "fmul v6.4s, v6.4s, v5.4s          \n"
        "fadd v6.4s, v6.4s, v0.4s          \n"
        "fadd v6.4s, v6.4s, v31.4s         \n"

        "fcvtzs v7.4s, v1.4s               \n"
        "add v7.4s, v7.4s, v20.4s          \n"
        "shl v7.4s, v7.4s, #23             \n"

        "fmul v6.4s, v6.4s, v7.4s          \n"
        "str  q6, [%[dst_ptr]]             \n"
      : [src_ptr] "+r" (src_ptr),
        [dst_ptr] "+r" (dst_ptr)
      : [vhalf] "r" (vhalf),
        [vcephes_exp_p0] "r" (vcephes_exp_p0)
      : "v0", "v1", "v2", "v3",
        "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11",
        "memory"
    );
  }

  for(; p < num; ++p) {
    dst[p] = exp(src[p]);
  }
}

#endif
