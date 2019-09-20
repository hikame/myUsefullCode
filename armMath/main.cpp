#include "armMath.hpp"
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>


int main() {
    int count = 10000;
    std::vector<float> data(count);
    std::vector<float> ref(count);
    std::vector<float> res_asm(count);

    for(int i = 0; i < count; ++i) {
      data[i] = (i % 10) / 10.f - 0.5f;
    }

    for(int i = 0; i < count; ++i) {
      ref[i] = exp(data[i]);
    }

     pengcuo_exp(data.data(), res_asm.data(), count);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < count; ++i) {
      ref[i] = exp(data[i]);
    }
    double time_std = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now()
                      - start).count();

    start = std::chrono::high_resolution_clock::now();
    pengcuo_exp(data.data(), res_asm.data(), count);
    double time_asm = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now()
                      - start).count();

    double relative_diff = 0.f;
    for(int i = 0; i < count; ++i) {

      // std::cout << i << " " << " " << data[i] << " " << ref[i] << "  "
      // << res_neon[i] << " "
      // << res_asm[i] << std::endl;

      relative_diff += fabs(res_asm[i] - ref[i]) / ref[i];
    }

    std::cout << "Compute " << count << " Element's EXP" << std::endl;
    std::cout << "Total Relative Diff : " << std::setw(8) << relative_diff << std::endl;
    std::cout << "     STD EXP        : " << std::setw(8) << time_std << " ms" << std::endl;
    std::cout << "     ASM            : " << std::setw(8) << time_asm << " ms"  << std::endl;
    return 0;
}
