#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "utils.hpp"
#include "cpu_calculator.hpp"
#include "gpu_calculator.hpp"
#include "kernel.h"

unsigned char alphabet[] = {
  'A', 'C', 'G', 'T'
};
const int alphabetLength = sizeof(alphabet);

int main(int argc, const char** argv) {
  struct timeval startc, end;
  long seconds, useconds;
  double mtime;
  bool isDMatricesIdentical = true, verbose = false, isCuda = false;

  if(argc < 3) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  levcuda [target string filepath] [source string filepath] (-v)" << std::endl;
    std::cout << "  -v: verbose" << std::endl;
    return EXIT_FAILURE;
  }

  if(argc > 3 && strcmp(argv[3], "-v") == 0) verbose = true;

  std::string s1 = read_file(argv[1]);
  std::string s2 = read_file(argv[2]);

  if(verbose) {
    std::cout << "Target string: " << s1 << std::endl;
    std::cout << "Source string: " << s2 << std::endl;
    std::cout << std::endl;
  }

  std::cout << "CPU calculation in progress" << std::endl;
  
  CpuCalculator calc(s1, s2);
  gettimeofday(&startc, NULL);
  calc.Calculate();
  gettimeofday(&end, NULL);

  seconds  = end.tv_sec  - startc.tv_sec;
  useconds = end.tv_usec - startc.tv_usec;
  mtime = useconds;
  mtime/=1000;
  mtime+=seconds*1000;

  std::cout << "CPU ms: " << mtime << std::endl;

  if(verbose) calc.Print();
  std::vector<std::string> transformationsStrings = calc.GetTransformations();
  save_edits_to_file(transformationsStrings, "./cpu_transformations.txt");

  // Gpu calculation if Gpu available
  if (isCuda = (cudaSetDevice(0) == cudaSuccess)) {
    std::cout << std::endl << "GPU calculation in progress" << std::endl;
    GpuCalculator gpuCalc(s1, s2, alphabet, alphabetLength);

    float ms;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    gpuCalc.Calculate();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU ms: " << ms << std::endl;

    if(verbose) gpuCalc.Print();
    transformationsStrings = gpuCalc.GetTransformations();
    save_edits_to_file(transformationsStrings, "./gpu_transformations.txt");

    for(int i = 0; i <= s2.size(); ++i) {
      for(int j = 0; j <= s1.size(); ++j) {
        if(gpuCalc.dMatrix[i*(s1.size()+1) + j] != calc.dMatrix[i][j])
          isDMatricesIdentical= false;
      }
    }
  } else {
    std::cout << "Cuda device not found!" << std::endl;
  }

  std::cout << std::endl << "Distance: " << calc.dMatrix[s2.size()][s1.size()] << std::endl;

  if(isCuda)
    std::cout << "Identical distance matrices?: " << (isDMatricesIdentical ? "true" : "false") << std::endl;

  return EXIT_SUCCESS;
}
