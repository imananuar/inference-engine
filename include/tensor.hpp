#include <vector>

#ifndef NYAN_UTILS_HPP
#define NYAN_UTILS_HPP

int add(int a, int b);
std::vector<std::vector<float>> createTensor(int row, int col);
void display2D(int row, int col, std::vector<std::vector<float>> tensor);


#endif