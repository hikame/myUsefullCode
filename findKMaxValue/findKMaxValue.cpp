#include <stdio.h>
#include <stdlib.h>

int findkMaxValue(int* src, int len, int k) {
    int value = src[0];
    int i = 1;
    int idx = 0;
    int j = len - 1;
    while(i <= j) {
        while(src[j] <= value && j >= i) --j;
        if(j >= i) {
            src[idx] = src[j];
            idx = j;
            --j;
        }

        while(src[i] >= value && i <= j) ++i;
        if(i <= j) {
            src[idx] = src[i];
            idx = i;
            ++i;
        }
    }
    src[idx] = value;

    if(idx == k) return src[idx];
    if(idx < k)return findkMaxValue(src + idx + 1, len - idx - 1, k - idx - 1);
    if(idx > k)return findkMaxValue(src, idx + 1, k);
}

int main(int argc, char** argv) {
    int kth = atoi(argv[1]);
    int src[10] = {2, 3, 1, 4, 5, 3, 7, 8, 3, 12};

    int res = findkMaxValue(src, 10, kth);
    printf("%d max num is %d\n", kth, res);

    return 0;
}
