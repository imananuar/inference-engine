#include "tensor.hpp"
#include <vector>
#include <iostream>

int add(int a, int b) {
    return a + b;
}
std::vector<std::vector<float>> createTensor(int row, int col) {
    std::vector<std::vector<float>> tensor(row, std::vector<float>(col, 0.0f));
    return tensor;
}

void display2D(std::vector<std::vector<float>> *mat) {
    int row = (*mat).size();
    int col = (*mat)[0].size();
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << (*mat)[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

std::vector<std::vector<float>> addMat(
    std::vector<std::vector<float>> matA,  
    std::vector<std::vector<float>> matB
) {
    if (matA.size() != matB.size() && matA[0].size() != matB[0].size()) {
        std::cerr << "ERROR: Difference size in two matrices!" << std::endl;
    }
    int row = matA.size();
    int col = matA[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            matA[i][j] = matA[i][j] + matB[i][j];
        }
    }

    return matA;
}