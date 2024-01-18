#include "gpu_calculator.hpp"

void GpuCalculator::Calculate() {
  int xMatrixLength = (n + 1) * alphabetLength, dMatrixLength = (n + 1) * (m + 1);
  int *xMatrix = new int[xMatrixLength], *d_xMatrix, *d_dMatrix, *d_charToAlphIndex;
  char *d_s1, *d_s2, *d_alphabet;

  dMatrix = new int[dMatrixLength];

  // Allocate X and D matrix on device
  HANDLE_ERROR(cudaMalloc((void**) &d_xMatrix, sizeof(int) * xMatrixLength));
  HANDLE_ERROR(cudaMalloc((void**) &d_dMatrix, sizeof(int) * dMatrixLength));

  // Copy strings to device
  HANDLE_ERROR(cudaMalloc((void**) &d_s1, sizeof(char) * n));
  HANDLE_ERROR(cudaMemcpy(d_s1, s1.c_str(), sizeof(char) * n, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**) &d_s2, sizeof(char) * m));
  HANDLE_ERROR(cudaMemcpy(d_s2, s2.c_str(), sizeof(char) * m, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**) &d_alphabet, sizeof(char) * alphabetLength));
  HANDLE_ERROR(cudaMemcpy(d_alphabet, alphabet, sizeof(char) * alphabetLength, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void**) &d_charToAlphIndex, sizeof(int) * REMAP_ARRAY_LENGTH));
  HANDLE_ERROR(cudaMemcpy(d_charToAlphIndex, charToAlphabetIndex, sizeof(int) * REMAP_ARRAY_LENGTH, cudaMemcpyHostToDevice));

  // Calculate X matrix
  calculateXMatrix<<<1, alphabetLength>>>(d_xMatrix, d_s1, d_alphabet, n);
  HANDLE_ERROR(cudaDeviceSynchronize());

  int blockCount = ((n+1) % BLOCK_SIZE) > 0 ? (n+1)/BLOCK_SIZE + 1 : (n+1)/BLOCK_SIZE;
  for(int j = 0; j <= m; ++j) {
    calculateDMatrix<<<blockCount, BLOCK_SIZE>>>(d_dMatrix, d_xMatrix, d_s1, d_s2, d_charToAlphIndex, n, m, j);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  HANDLE_ERROR(cudaMemcpy(dMatrix, d_dMatrix, sizeof(int) * dMatrixLength, cudaMemcpyDeviceToHost));

  // Clean allocated device's memory
  HANDLE_ERROR(cudaFree(d_xMatrix));
  HANDLE_ERROR(cudaFree(d_dMatrix));
  HANDLE_ERROR(cudaFree(d_s1));
  HANDLE_ERROR(cudaFree(d_s2));
  HANDLE_ERROR(cudaFree(d_alphabet));
  HANDLE_ERROR(cudaFree(d_charToAlphIndex));
} 

std::vector<std::string> GpuCalculator::GetTransformations() {
    std::vector<std::string> transformations = std::vector<std::string>();
    int i = m, j = n;

    while(i > 0 || j > 0) {
        if (i > 0 && dMatrix[i*(n+1) + j] == dMatrix[(i-1)*(n+1) + j] + 1) {
            transformations.push_back("Delete "+ std::string(1,s2[i - 1]) + " at " + std::to_string(i - 1));
            i--;
        }
        else if(j > 0 && dMatrix[i*(n+1) + j] == dMatrix[i*(n+1) + (j-1)] + 1) {
            transformations.push_back("Insert "+ std::string(1,s1[j - 1]) + " at " + std::to_string(j - 1));
            j--;
        }
        else {
            if(i > 0 && j > 0 && dMatrix[i*(n+1) + j] == dMatrix[(i-1)*(n+1) + (j-1)] + 1) {
                if(s1[i - 1] != s2[j - 1]) {
                    transformations.push_back("Substitute "+ std::string(1,s2[i - 1]) + " at " + std::to_string(i - 1)
                     + " with " + std::string(1,s2[j - 1]));
                }   
            }
            i--;
            j--;
        }
    }
    return transformations;
} 

void GpuCalculator::Print() {
  for(int i = 0; i <= m; ++i) {
    for(int j = 0; j <= n; ++j) {
      std::cout << dMatrix[i*(n+1) + j] << " ";
    }
    std::cout << std::endl;
  }
} 
