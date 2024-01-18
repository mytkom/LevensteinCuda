#include "cpu_calculator.hpp"

void CpuCalculator::Calculate() {
  int cost = 0;
  dMatrix = std::vector<std::vector<int>>(m + 1, std::vector<int>(n + 1, 0));

  for(int i = 0; i <= n; ++i)
    dMatrix[0][i] = i;

  for(int i = 0; i <= m; ++i)
    dMatrix[i][0] = i;

  for(int j = 1; j <= m; ++j) {
    for(int i = 1; i <= n; ++i) {
      if(s1[i-1] != s2[j-1])
        cost = 1;
      else
        cost = 0;

      dMatrix[j][i] += std::min({
          dMatrix[j-1][i-1] + cost,
          dMatrix[j][i-1] + 1,
          dMatrix[j-1][i] + 1
          });
    }
  }  
} 

std::vector<std::string> CpuCalculator::GetTransformations() {
    std::vector<std::string> list = std::vector<std::string>();
    int i = m, j = n;

    while(i > 0 || j > 0) {
        if (i > 0 && dMatrix[i][j] == dMatrix[i - 1][j] + 1) {
            list.push_back(std::string("Delete "+ std::string(1,s2[i - 1]) + " at " + std::to_string(i - 1)));
            i--;
        }
        else if(j > 0 && dMatrix[i][j] == dMatrix[i][j - 1] + 1) {
            list.push_back(std::string("Insert "+ std::string(1,s1[j - 1]) + " at " + std::to_string(j - 1)));
            j--;
        }
        else {
            if(i > 0 && j > 0 && dMatrix[i][j] == dMatrix[i - 1][j - 1] + 1) {
                if(s1[i - 1] != s2[j - 1]) {
                    list.push_back(std::string("Substitute "+ std::string(1,s2[i - 1]) + " at " + std::to_string(i - 1)
                     + " with " + std::string(1,s2[j - 1])));
                }   
            }
            i--;
            j--;
        }
    }
    return list;
} 

void CpuCalculator::Print() {
  std::for_each(dMatrix.cbegin(), dMatrix.cend(), [](const std::vector<int> &row){
    std::for_each(row.cbegin(), row.cend(), [](const int &value) { std::cout << value << " "; });
    std::cout << std::endl;
  });
} 
