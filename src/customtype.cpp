#include "customtype.hpp"

// SINGLE HIDDEN LAYER
// Describing how many nodes are that in particular layer
// Param: Nodes in the layer, input size (number of nodes before)
// THIS IS NOT A ARRAY OF LAYER.

HiddenLayer::HiddenLayer(int nodes, int inputSize)
{
    // Assuming input: (2, 3)
    this->nodes = nodes;

    // BIAS: An array of scalar
    // Array.length = number of nodes.
    // Initialization might be 0? Then we play around with the bias
    this->bias = cv::Mat::ones(nodes, 1, CV_32F) * 0.01f;

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
