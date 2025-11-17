#include <vector>
#include <opencv2/opencv.hpp>


#ifndef NYAN_UTILS_HPP
#define NYAN_UTILS_HPP

enum Operation {
    ADD,
    SUBTRACT
};

int add(int a, int b);
void display2D(std::vector<std::vector<float>> *mat);

std::vector<std::vector<float>> createTensor(int row, int col);
std::vector<std::vector<float>> addMat(Operation op, std::vector<std::vector<float>> matA, std::vector<std::vector<float>> matB);
std::vector<std::vector<float>> mulmat(std::vector<std::vector<float>> matA, std::vector<std::vector<float>> matB);

void activation_func(cv::Mat *sample_img, cv::Mat *weight, cv::Mat *bias, int layer);

#endif