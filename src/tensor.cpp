#include "tensor.hpp"
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <customtype.hpp>

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

void activation_func(cv::Mat *input, HiddenLayer layers)
{
    if (input == nullptr || input->empty())
    {
        std::cerr << "Error: Invalid image pointer or empty image." << std::endl;
        return;
    }

    std::cout << "\n\nNEURAL NETWORK FORWARD PROPAGATION START" << std::endl;
    std::cout << "TOTAL LAYER: " << layers.getNodes().size() << std::endl;
    
    cv::Mat x = *input;
    for (int i = 0; i < layers.getNodes().size(); i++)
    {
        int nodes = layers.getNodes().at(i);
        std::cout << "Running for layer: " << i << std::endl;
        std::cout << "Nodes size: " << nodes << std::endl;
        std::cout << "\n";
        
        cv::Mat z_mat = cv::Mat::zeros(nodes, 1, CV_32F);
        cv::Mat weight = layers.getWeight().at(i);
        cv::Mat bias = layers.getBias().at(i);

        std::cout << "Input: " << x << std::endl;
        std::cout << "\n";
        std::cout << "Weight: " << weight << std::endl;
        std::cout << "\n";
        std::cout << "Bias: " << bias << std::endl;
        
        // For each node
        for (int j = 0; j < nodes; j++)
        {
            float z;
            // Summation x*w
            std::cout << "\nNODE NO: " << j << std::endl;
            std::cout << "z = ";
            for (int row = 0; row < x.rows; row++) {
                z = (float)x.at<float>(row, 0) * (float)weight.at<float>(0, row);
                std::cout << (float)x.at<float>(row, 0) << " * " << (float)weight.at<float>(0, row) << " + ";
            }
            // Add bias
            std::cout << (float)bias.at<float>(j, 0) << std::endl;
            z += (float)bias.at<float>(j, 0);
            
            // Append to z matrix -> This will be new input for second layer
            z_mat.at<float>(j, 0) = z;
            std::cout << "z = " << z << std::endl;
            x = z;
        }

        std::cout << "\nZ_Matrix for layer: " << i << std::endl;
        std::cout << z_mat << std::endl;
        x = z_mat;
        std::cout << "\nNext Input: " << x << std::endl;
        std::cout << "----------------------------------------------------------\n" << std::endl;
    }
}