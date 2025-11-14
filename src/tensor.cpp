#include "tensor.hpp"
#include <vector>
#include <iostream>

int add(int a, int b) {
    return a + b;
}
std::vector<std::vector<float>> createTensor(int row, int col) {
    // std::vector<float> tensor(10, 0);
    std::vector<std::vector<float>> tensor(row, std::vector<float>(col, 0));
    display2D(row, col, tensor);

    return tensor;
}

void display2D(int row, int col, std::vector<std::vector<float>> tensor) {
        for (auto &row: tensor) {
        for (auto val: row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    };
}