#include "customtype.hpp"

// SINGLE HIDDEN LAYER
// Describing how many nodes are that in particular layer
// Param: Nodes in the layer, input size (number of nodes before)
// THIS IS NOT A ARRAY OF LAYER.

// HLayer::HLayer(int in, int out)
// {
//     this->layer.in = in;
//     this->layer.out = out;
//     this->layer.b = std::vector<float>(out, 0.01f);
// }

HiddenLayer::HiddenLayer(int nodes, int inputSize)
{
    // Assuming input: (2, 3)
    this->nodes = nodes;

    // BIAS: An array of scalar
    // Array.length = number of nodes.
    // Initialization might be 0? Then we play around with the bias
    this->bias = cv::Mat::ones(nodes, 1, CV_32F) * 0.01f;
    this->part_dL_db = cv::Mat::zeros(nodes, 1, CV_32F);

    for (int i = 0; i < nodes; i++)
    {
        // WEIGHT: An array of matrix
        // Array.length = number of nodes in the hidden layer
        // Matrix.length = Number of input nodes
        // Initialization using Kaiming Init. Src: https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/
        cv::Mat w = cv::Mat::zeros(1, inputSize, CV_32F);
        float stddev = std::sqrt( 2.0f / inputSize );
        cv::randn(w, 0, stddev);
        this->weight.push_back(w);
    }
}

int HiddenLayer::getNodes() {
    return nodes;
}

std::vector<cv::Mat> HiddenLayer::getWeight() {
    return weight;
}

cv::Mat HiddenLayer::getBias() {
    return bias;
}

void HiddenLayer::displayWeight()
{
    for (int i = 0; i < weight.size(); i++ )
    {
        std::cout << "\nWeight for node = " << i+1 << std::endl;
        std::cout << weight.at(i) << std::endl;
    }
}

void HiddenLayer::displayBias()
{
    std::cout << "\nBias:" << std::endl;
    std::cout << bias << std::endl;
}

void HiddenLayer::setWeightedSum(float w)
{
    this->w_sum.push_back(w);
}

cv::Mat HiddenLayer::getWeightedSum()
{
    return w_sum;
}

void HiddenLayer::setActivation(float z)
{
    // this->activation.at<float>(row, 0) = z;
    this->activation.push_back(z);
}

cv::Mat HiddenLayer::getActivation()
{
    return activation;
}

void HiddenLayer::setUpdatedBias(int row, float bias_val)
{
    this->part_dL_db.at<float>(row, 0) = bias_val;
}

cv::Mat HiddenLayer::getUpdatedBias()
{
    return part_dL_db;
}

void HiddenLayer::setTotalUpdatedBias(cv::Mat updatedBias)
{
    this->dL_dB.push_back(updatedBias);
}

std::vector<cv::Mat> HiddenLayer::getTotalUpdatedBias()
{
    return dL_dB;
}
