#include <iostream>
#include "tensor.hpp"
#include <vector>

int main(int argc, char* argv[]) {
    std::cout << "Running Inference Engine" << std::endl;
    std::cout << "Testing .hpp file: " << add(1,5) << std::endl;
    int row = 3;
    int col = 3;
    std::vector<std::vector<float>> matA = createTensor(row, col);
    matA[0][0] = 1;
    matA[0][1] = 1;
    matA[0][2] = 1;

    matA[1][0] = 1;
    matA[1][1] = 1;
    matA[1][2] = 1;
       
    matA[2][0] = 1;
    matA[2][1] = 1;
    matA[2][2] = 1;
    display2D(&matA);
    
    std::vector<std::vector<float>> matB = createTensor(row, col);
    matB[0][0] = 4;
    matB[1][0] = 2;
    matB[2][0] = 3;

    matB[0][1] = 5;
    matB[1][1] = 1;
    matB[2][1] = 2;

    // matB[0][2] = 1;
    // matB[1][2] = 4;
    // matB[2][2] = 5;
    display2D(&matB);

    // std::vector<std::vector<float>> sumMat = addMat(SUBTRACT, matA, matB);
    // display2D(&sumMat);

    std::vector<std::vector<float>> mul = mulmat(matA, matB);
    display2D(&mul);

    return 0;
}