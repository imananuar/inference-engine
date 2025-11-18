#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <customtype.hpp>

enum Operation {
    ADD,
    SUBTRACT
};

int add(int a, int b);
void display2D(std::vector<std::vector<float>> *mat);

std::vector<std::vector<float>> createTensor(int row, int col);
std::vector<std::vector<float>> addMat(Operation op, std::vector<std::vector<float>> matA, std::vector<std::vector<float>> matB);
std::vector<std::vector<float>> mulmat(std::vector<std::vector<float>> matA, std::vector<std::vector<float>> matB);

cv::Mat activation_func(cv::Mat *input, HiddenLayer hiddenLayer);
cv::Mat softmax_func(cv::Mat output);