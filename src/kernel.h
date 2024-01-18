#pragma once

#include "gpu_calculator.hpp"
#define FULL_MASK 0xffffffff
#define THREADS_IN_WARP 32

__global__ void calculateXMatrix(int *xMatrix, const char *s, const char *alphabet, int n);
__global__ void calculateDMatrix(int *dMatrix, int *xMatrix, const char *s1, const char *s2, const int *charToAlphIndex, int n, int m, int j);
