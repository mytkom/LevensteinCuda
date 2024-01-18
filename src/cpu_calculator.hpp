#pragma once

#include "calculator.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

class CpuCalculator : Calculator {
  public:
    std::vector<std::vector<int>> dMatrix;
    CpuCalculator(const std::string &s1, const std::string &s2) : Calculator(s1, s2) { }
    void Calculate(); 
    std::vector<std::string> GetTransformations(); 
    void Print(); 
};
