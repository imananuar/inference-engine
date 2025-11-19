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

void train_model(std::vector<int> layers, cv::Mat input)
{
    std::cout << "\nTRAINING MODEL -> " << layers.size() << " Layers\n" << std::endl;
    std::vector<HiddenLayer> hiddenLayer_arr;
    
    for (int nodes: layers)
    {
        cv::Mat a_mat;
        int inputSize = input.rows * input.cols;
        HiddenLayer HiddenLayer(nodes, inputSize);
        HiddenLayer.displayWeight();
        HiddenLayer.displayBias();
        hiddenLayer_arr.push_back(HiddenLayer);
        for (int node = 0; node < nodes; node++)
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

            // We only keep the a value because that is only matter to us
            a_mat.push_back(a);

        }
        std::cout << "\nResult for node " << nodes << ":" << std::endl;
        std::cout << a_mat << std::endl;
    }
}

// cv::Mat activation_func(cv::Mat *input, HiddenLayer layers)
// {
//     if (input == nullptr || input->empty())
//     {
//         std::cerr << "Error: Invalid image pointer or empty image." << std::endl;
//         return cv::Mat();
//     }

//     std::cout << "\n\nNEURAL NETWORK FORWARD PROPAGATION START" << std::endl;
//     std::cout << "TOTAL LAYER: " << layers.getNodes().size() << std::endl;
    
//     cv::Mat x = *input;
//     for (int i = 0; i < layers.getNodes().size(); i++)
//     {
//         // int nodes = layers.getNodes().at(i);
//         std::cout << "Running for layer: " << i << std::endl;
//         std::cout << "Nodes size: " << nodes << std::endl;
//         std::cout << "\n";
        
//         cv::Mat z_mat = cv::Mat::zeros(nodes, 1, CV_32F);
//         cv::Mat weight = layers.getWeight().at(i);
//         // cv::Mat bias = layers.getBias().at(i);

//         std::cout << "Input: " << x << std::endl;
//         std::cout << "\n";
//         std::cout << "Weight: " << weight << std::endl;
//         std::cout << "\n";
//         std::cout << "Bias: " << bias << std::endl;
        
//         // For each node
//         for (int j = 0; j < nodes; j++)
//         {
//             float z;
//             // Summation x*w
//             // std::cout << "\nNODE NO: " << j << std::endl;
//             // std::cout << "z = ";
//             for (int row = 0; row < x.rows; row++) {
//                 z = (float)x.at<float>(row, 0) * (float)weight.at<float>(0, row);
//                 std::cout << (float)x.at<float>(row, 0) << " * " << (float)weight.at<float>(0, row) << " + ";
//             }
//             // Add bias
//             std::cout << (float)bias.at<float>(j, 0) << std::endl;
//             z += (float)bias.at<float>(j, 0);
            
//             // Append to z matrix -> This will be new input for second layer
//             float a = std::max(0.00000000000f, z);
//             z_mat.at<float>(j, 0) = a;

//             std::cout << "z = " << z << std::endl;
//             std::cout << "a = " << a << std::endl;

//             x = z;
//         }

//         x = z_mat;
//         // if (i == layers.getNodes().size()-1) { return x; }
//         std::cout << "\nZ_Matrix for layer: " << i << std::endl;
//         std::cout << z_mat << std::endl;
//         std::cout << "\nNext Input: " << x << std::endl;
//         std::cout << "----------------------------------------------------------\n" << std::endl;
//     }
//     return cv::Mat();
// }

// cv::Mat softmax_func(cv::Mat output)
// {
//     // Formula is e^z/∑e^z
//     // For more detail: https://www.datacamp.com/tutorial/softmax-activation-function-in-python?utm_cid=19589720824&utm_aid=157156376311&utm_campaign=230119_1-ps-other~dsa~tofu_2-b2c_3-apac_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9066735-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~apac-en~dsa~tofu~tutorial~python&gad_source=1&gad_campaignid=19589720824&gbraid=0AAAAADQ9WsHcgp7Oz8BWcxEEYUE6DKIob&gclid=Cj0KCQiArOvIBhDLARIsAPwJXOZytP5zvCvDflsRxJISDB1T4qvQAsfdiJQWUacJ4Xb-efZ2NU6sAeoaAkZyEALw_wcB

//     cv::Mat y_hat = cv::Mat::zeros(output.rows, 1, CV_32F);
//     float sum_exp = 0.0f;
//     for (int i = 0; i < output.rows; i++)
//     {
//         sum_exp += std::exp(output.at<float>(i, 0));
//     }

//     for (int i = 0; i < output.rows; i++)
//     {
//         y_hat.at<float>(i,0) = (output.at<float>(i,0) / sum_exp);
//     }

//     std::cout << "y_hat: " << y_hat << std::endl;
    
//     return y_hat;
// }