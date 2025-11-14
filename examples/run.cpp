#include <iostream>
#include "tensor.hpp"
#include <vector>

int main(int argc, char* argv[]) {
    std::cout << "Running Inference Engine" << std::endl;
    std::cout << "Testing .hpp file: " << add(1,5) << std::endl;
    int row = 3;
    int col = 3;
    std::vector<std::vector<float>> matA = createTensor(row, col);
    matA[2][2] = 2;
    matA[0][1] = 5;
    display2D(&matA);
    
    std::vector<std::vector<float>> matB = createTensor(row, col);
    matB[0][1] = 1;
    matB[2][2] = 10;
    display2D(&matB);

    std::vector<std::vector<float>> sumMat = addMat(SUBTRACT, matA, matB);
    display2D(&sumMat);

    return 0;
}