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

void train_model(std::vector<cv::Mat> inputMat[], cv::Mat inputAns, int epoch, std::vector<int> layers, float16_t learning_rate)
{
    int batchSize = inputAns.rows;
    int roundSize = batchSize / epoch;
    std::cout << inputMat->size() << std::endl;
    std::cout << "\nTRAINING MODEL" << std::endl;
    std::cout << "Input Matrix [col, row]: " << inputMat->at(0).size() << std::endl;
    std::cout << "Training Size: " << batchSize << std::endl;
    std::cout << "Layers: " << layers.size() << std::endl;
    std::cout << "Round Size: " << roundSize << std::endl;
    

    std::vector<HiddenLayer> hiddenLayer_arr;
    
    cv::Mat raw_y;
    cv::Mat y_hats;
    cv::Mat ce_loss;
    float eSum;

    for (int i = 0; i < roundSize; i++)
    {
        std::cout << "Training for Round " << i << std::endl;
        cv::Mat input = inputMat->at(i);
        for (int layer = 0; layer < layers.size(); layer++)
        {
            cv::Mat a_mat;
            int inputSize = input.rows * input.cols;
            HiddenLayer HiddenLayer(layers[layer], inputSize);
            std::cout << "Processing for layer: " << layer + 1;
            std::cout << ", Node size:  " << layers[layer] << std::endl;

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
                a = std::max(0.01f, z);

                // Calculate softmax denominator when reach last layer = output layer
                if ( layer == layers.size() - 1 ) {
                    std::cout << "Last layer with node = " << layers[layer] << std::endl; 
                    eSum += std::exp(a);
                }

                // We only keep the a value because that is only matter to us
                a_mat.push_back(a);
            }
    
            if ( layer == layers.size()-1 )
            {
                raw_y = a_mat;
                for (int y = 0; y < raw_y.rows; y++)
                {
                    float32_t y_hat = std::exp(raw_y.at<float>(y, 0)) / eSum;
                    std::cout << "std::exp(raw_y.at<float>(y, 0)) = " << std::exp(raw_y.at<float>(y, 0)) << std::endl;
                    std::cout << "eSum = " << eSum << std::endl;
                    std::cout << "y_hat = " << y_hat << std::endl;
                    y_hats.push_back(y_hat);

                    float32_t ce = -1.0f * std::log(y_hat);
                    ce_loss.push_back(ce);
                }
            }
        }
    
        std::cout << "\nTRAINING COMPLETED!" << std::endl;
        std::cout << "Output Node (Before softmax function)" << std::endl;
        std::cout << raw_y << std::endl;
        std::cout << "\ny_hat (After apply softmax)" << std::endl;
        std::cout << y_hats << std::endl;
    
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(y_hats, nullptr, &maxVal, nullptr, &maxLoc);
    
        int real_y = 1;
        int predict_y = maxLoc.y;
        std::cout << "\nCross Entropy Loss: " << std::endl;
        std::cout << ce_loss << std::endl;
        std::cout << "\nTotal Cross Entropy: " << cv::sum(ce_loss)[0] << std::endl;
        
        std::cout << "\nObserved value = " << inputAns.at<int32_t>(i, 0) << std::endl;
        std::cout << "Predicted value = " << predict_y << std::endl;
        break;
    }

    std::cout << "\n" << std::endl;
}