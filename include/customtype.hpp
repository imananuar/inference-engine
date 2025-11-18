#pragma once
#include <opencv2/opencv.hpp>

struct DrawImage
{
    cv::Point start_point;
    cv::Point end_point;
    bool drawing = false;
    cv::Mat image;
};

class HiddenLayer
{
    private:
    std::vector<int> nodes;
    std::vector<cv::Mat> bias;
    std::vector<cv::Mat> weight;

    public:
    HiddenLayer(std::vector<int> nodes, int inputSize);

    std::vector<int> getNodes();

    std::vector<cv::Mat> getWeight();
    std::vector<cv::Mat> getBias();
};