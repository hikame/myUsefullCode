#include <stdio.h>
bool isLittleEndian() {
    int a = 0x1234;
    char b = *(char*) &a;
    if(0x34 == b) {
        return true;
    }
    else {
        return false;
    }
}

int main() {
    bool flag = isLittleEndian();
    if(flag) {
        printf("isLittleEndian\n");
    }
    else {
        printf("is not LittleEndian");
    }
    return 0;
}
