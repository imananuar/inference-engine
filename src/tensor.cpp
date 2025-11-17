#include "tensor.hpp"
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

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
    Operation op,
    std::vector<std::vector<float>> matA,  
    std::vector<std::vector<float>> matB
) {
    if (matA.size() != matB.size() && matA[0].size() != matB[0].size()) 
    {
        std::cerr << "ERROR: Difference size in two matrices!" << std::endl;
    }
    int row = matA.size();
    int col = matA[0].size();

    for (int i = 0; i < row; i++) 
    {
        for (int j = 0; j < row; j++) 
        {
            if (op == ADD) { matA[i][j] = matA[i][j] + matB[i][j]; }
            else { matA[i][j] = matA[i][j] - matB[i][j]; }
        }
    }

    return matA;
}

// Making matrics multiplication from scratch
std::vector<std::vector<float>> mulmat(
    std::vector<std::vector<float>> matA,
    std::vector<std::vector<float>> matB
) {
    int row = matA.size();
    int col = matA[0].size();

    std::vector<std::vector<float>> matC = createTensor(row, col);

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            for (int k = 0; k < col; k++)
            {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }

    return matC;
}

void activation_func(cv::Mat *input, cv::Mat *weight, cv::Mat *bias, int nodes)
{
    if (input == nullptr || input->empty())
    {
        std::cerr << "Error: Invalid image pointer or empty image." << std::endl;
        return;
    }
    
    cv::Mat z_mat = cv::Mat::zeros(nodes, 1, CV_32F);
    for (int n = 0; n < nodes; n++)
    {
        float z;
        for (int r = 0; r < input->rows; r++) 
        {
            for (int c = 0; c < input->cols; c++) 
            {
                z = (float)input->at<float>(r, c) * (float)weight->at<float>(c, r);
            }
        }
        z += (float)weight->at<float>(0, n);
        z_mat.at<float>(n, 0) = z;
    }

    std::cout << z_mat << std::endl;
}