#pragma once

#include <vector>
#include <string>

class Calculator {
  public:
    Calculator(const std::string &s1, const std::string &s2) {
      this->s1 = s1;
      this->s2 = s2;
      n = s1.size();
      m = s2.size();
    }
    virtual void Calculate() = 0; 
    virtual std::vector<std::string> GetTransformations() = 0; 
    virtual void Print() = 0; 

  protected:
    std::string s1, s2;
    int n, m;
};
