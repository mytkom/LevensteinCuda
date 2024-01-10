#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

std::string s1 = "kitten";
std::string s2 = "sitting";

void print(const std::vector<std::vector<int>> &distanceMatrix);
void calculate(std::vector<std::vector<int>> &distanceMatrix);

int main(int argc, const char** argv) {
  int n = size(s1), m = size(s2);
  std::vector<std::vector<int>> distanceMatrix(n + 1, std::vector<int>(m + 1, 0));

  for(int i = 0; i <= m; ++i)
    distanceMatrix[0][i] = i;

  for(int i = 0; i <= n; ++i)
    distanceMatrix[i][0] = i;

  calculate(distanceMatrix);

  std::cout << "Distance: " << distanceMatrix[n][m] << std::endl;

  print(distanceMatrix);

  return 0;
}

void print(const std::vector<std::vector<int>> &distanceMatrix) {
  std::for_each(distanceMatrix.cbegin(), distanceMatrix.cend(), [](const std::vector<int> &row){
      std::for_each(row.cbegin(), row.cend(), [](const int &value) { std::cout << value << " "; });
      std::cout << std::endl;
  });
}

void calculate(std::vector<std::vector<int>> &distanceMatrix) {
  int cost = 0, n = distanceMatrix.size(), m = distanceMatrix[0].size();

  for(int i = 1; i < n; ++i) {
    for(int j = 1; j < m; ++j) {
      if(i == 1 || j == 1 || s1[i-1] != s2[j-1])
        cost = 1;
      else
        cost = 0;

      distanceMatrix[i][j] += std::min({
          distanceMatrix[i-1][j-1] + cost,
          distanceMatrix[i][j-1] + 1,
          distanceMatrix[i-1][j] + 1
      });
    }
  }  
}
