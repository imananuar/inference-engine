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
    int nodes;
    std::vector<cv::Mat> weight;
    cv::Mat bias;

    public:
    HiddenLayer(int nodes, int inputSize);

    int getNodes();
    std::vector<cv::Mat> getWeight();
    cv::Mat getBias();
    void displayWeight();
    void displayBias();
};