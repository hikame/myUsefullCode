__kernel void sgemm_mult_only(     
        __global const float*  A,       
        const int              lda,     
        __global float*        C,       
        const int              ldc,     
        const int              m,       
        const int              n,       
        const int              k,       
        __read_only image2d_t  Bi){     
    int gx = get_global_id(0);   
    int gy = get_global_id(1);   
    if(((gx << 2) < n) && ((gy << 3) < m)) {
        float4 a[8]; 
        float4 b[4]; 
        float4 c[8]; 
        for(int i = 0; i < 8; ++i) { 
            c[i] = 0.f; 
        }
        int A_y_off = (gy << 3) * lda; 
        for(int pos = 0; pos < k; pos += 4) {
#pragma unroll 
            for(int i = 0; i < 4; ++i) { 
                b[i] = read_imagef(Bi, (int2)(gx, pos +i)); 
            } 
            int A_off = A_y_off + pos; 
#pragma unroll 
            for(int i = 0; i < 8; ++i) { 
                a[i] = vload4(0, A + A_off); 
                A_off += lda; 
            } 
#pragma unroll 
            for(int i = 0; i < 8; ++i) {
                c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3]; 
            } 
        } 
#pragma unroll 
        for(int i = 0; i < 8; ++i) { 
            int C_offs = (((gy << 3) + i ) * ldc + (gx << 2)); 
            vstore4(c[i], 0, C + C_offs); 
        } 
    } 
} 
