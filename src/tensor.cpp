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

void train_model(std::vector<int> layers, cv::Mat input, float16_t learning_rate)
{
    std::cout << "\nTRAINING MODEL -> " << layers.size() << " Layers\n" << std::endl;
    std::vector<HiddenLayer> hiddenLayer_arr;
    
    cv::Mat raw_y;
    cv::Mat y_hat;
    float eSum;
    for (int layer = 0; layer < layers.size(); layer++)
    {
        cv::Mat a_mat;
        int inputSize = input.rows * input.cols;
        HiddenLayer HiddenLayer(layers[layer], inputSize);
        // HiddenLayer.displayWeight();
        // HiddenLayer.displayBias();
        hiddenLayer_arr.push_back(HiddenLayer);
        for (int node = 0; node < layers[layer]; node++)
        {
            // For each node, we will find x, y value for that node
            // x = z = ∑input*w + b
            float z;
            float a;
            for (int row = 0; row < input.rows; row++)
            {
                z += input.at<float>(row, 0) * HiddenLayer.getWeight().at(node).at<float>(0, row);
            }
            z += HiddenLayer.getBias().at<float>(node, 0);

            // y = a = ReLU(z);
            a = std::max(0.0000000f, z);
            eSum += std::exp(a);
            // We only keep the a value because that is only matter to us
            a_mat.push_back(a);
        }

        if ( layer == layers.size()-1 )
        {
            raw_y = a_mat;
            for (int y = 0; y < raw_y.rows; y++)
            {
                y_hat.push_back(std::exp(raw_y.at<float>(y, 0)) / eSum);
            }
        }
    }

    std::cout << "\nTraining done!\n" << std::endl;
    std::cout << "Output Node (Before softmax function)" << std::endl;
    std::cout << raw_y << std::endl;
    std::cout << "\ny_hat (After apply softmax)" << std::endl;
    std::cout << y_hat << std::endl;

    double maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(y_hat, nullptr, &maxVal, nullptr, &maxLoc);
    std::cout << "\nAnswer is: " << maxLoc.y << std::endl;
}