#include <vector>

#ifndef NYAN_UTILS_HPP
#define NYAN_UTILS_HPP

int add(int a, int b);
void display2D(std::vector<std::vector<float>> *mat);

std::vector<std::vector<float>> createTensor(int row, int col);
std::vector<std::vector<float>> addMat(std::vector<std::vector<float>> matA, std::vector<std::vector<float>> matB);


#endif