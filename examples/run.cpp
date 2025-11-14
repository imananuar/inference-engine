#include <iostream>
#include "tensor.hpp"
#include <vector>

int main(int argc, char* argv[]) {
    std::cout << "Running Inference Engine" << std::endl;
    std::cout << "Testing .hpp file: " << add(1,5) << std::endl;
    createTensor(3,3);
    return 0;
}