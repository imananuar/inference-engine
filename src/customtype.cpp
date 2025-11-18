#include "customtype.hpp"

HiddenLayer::HiddenLayer(std::vector<int> nodes, int inputSize)
{
    this->nodes = nodes;

    for (int i = 0; i < nodes.size(); i++)
    {
        // weight matrix: (nodes[i] x inputSize)
        cv::Mat w = cv::Mat::zeros(1, inputSize, CV_32F);
        cv::randu(w, -0.1f, 0.1f);

        // bias: (nodes[i] x 1)
        cv::Mat b = cv::Mat::zeros(nodes[i], 1, CV_32F);
        cv::randu(b, -0.1f, 0.1f);

        weight.push_back(w);
        bias.push_back(b);

        inputSize = nodes[i];
    }
}

std::vector<int> HiddenLayer::getNodes() {
    return nodes;
}

std::vector<cv::Mat> HiddenLayer::getWeight() {
    return weight;
}

std::vector<cv::Mat> HiddenLayer::getBias() {
    return bias;
}
