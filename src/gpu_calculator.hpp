#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include "cuda.h"
#include "cuda_runtime.h"
#include "kernel.h"
#include "calculator.hpp"

#ifndef HANDLE_ERROR

static void HandleError(cudaError_t err, const char* file, int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
        file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

#endif

#define REMAP_ARRAY_LENGTH 256
#define BLOCK_SIZE 512

class GpuCalculator : Calculator {
  public:
    int *dMatrix;
    GpuCalculator(const std::string &s1, const std::string &s2, std::string &alphabet) : Calculator(s1, s2) {
      this->alphabet = alphabet;

      for(int i = 0; i < alphabet.length(); ++i)
        charToAlphabetIndex[alphabet[i]] = i; 
    }
    void Calculate(); 
    std::vector<std::string> GetTransformations(); 
    void Print(); 

  private:
    int charToAlphabetIndex[REMAP_ARRAY_LENGTH];
    std::string alphabet;
};
