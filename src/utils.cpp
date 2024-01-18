#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <ostream>

std::string read_file(const std::string& path) {
    
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cout<<"Error opening file: " << path << std::endl;
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(input_file)),
                        std::istreambuf_iterator<char>());

    input_file.close();

    if(content[content.length()-1] == 10)
      content.pop_back();

    return content;
}


void save_edits_to_file(const std::vector<std::string>& edits, const std::string& file_name) {     
    
    std::ofstream file(file_name);

    for(const auto& edit: edits) {
        file << edit << '\n';
    }

    file.close();
} 
