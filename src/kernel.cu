#include "kernel.h"
#include <cstdio>

__global__ void calculateXMatrix(int *xMatrix, const char *s, const char *alphabet, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int prev = 0;
  char rowChar = alphabet[i];

  for(int j = 0; j <= n; ++j) {
    if(j > 0  && s[j-1] == rowChar) {
      xMatrix[i*(n+1) + j] = j;
      prev = j;
    } else {
      xMatrix[i*(n+1) + j] = prev;
    }
  }
}

__global__ void calculateDMatrix(int *dMatrix, int *xMatrix, const char *s1, const char *s2, const int *charToAlphIndex, int n, int m, int j) {
  __shared__ int interWarpMemory[BLOCK_SIZE/THREADS_IN_WARP-1];

  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + tid;
  int l, tempX, Avar, Bvar, Cvar, Dvar;

  if(i > n) return;

  if(j > 0) {
    // get Bvar (previous Dvar) from global memory
    Bvar = dMatrix[(j-1)*(n+1) + i];

    // get Alphabet array index
    l = charToAlphIndex[s2[j-1]];

    // intra warp Bvar obtaining
    Avar = __shfl_up_sync(FULL_MASK, Bvar, 1);
  }

  // if last thread - save Bvar to shared memory
  if(tid % THREADS_IN_WARP == THREADS_IN_WARP - 1) {
    interWarpMemory[tid/THREADS_IN_WARP] = Bvar;
  }

  // make sure all warps in block has written to shared memory
  __syncthreads();

  if(tid == 0 && blockIdx.x != 0) {
    // if first thread of not first block - get Avar from global memory
    Avar = dMatrix[(j-1)*(n+1) + (i-1)];
  } else if(tid % THREADS_IN_WARP == 0 && tid > 0) {
    // if first thread of not first warp - get Avar from shared memory
    Avar = interWarpMemory[tid/THREADS_IN_WARP - 1];
  }

  // get X[l,i]
  tempX = xMatrix[l*(n+1) + i];

  // calculate Dvar
  if(j == 0) {
    Dvar = i;
  } else if(i == 0) {
    Dvar = j;
  } else if(s1[i-1] == s2[j-1]) {
    Dvar = Avar;
  } else if(tempX == 0) {
    Dvar = 1 + min(Avar, min(Bvar, i+j-1));
  } else {
    Cvar = dMatrix[(j-1)*(n+1) + (tempX-1)];
    Dvar = 1 + min(Avar, min(Bvar, Cvar + (i-1-tempX)));
  }

  // save result to global memory
  dMatrix[j*(n+1) + i] = Dvar;
}

