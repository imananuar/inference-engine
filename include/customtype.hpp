#pragma once
#include <opencv2/opencv.hpp>

struct DrawImage
{
    cv::Point start_point;
    cv::Point end_point;
    bool drawing = false;
    cv::Mat image;
};

struct Layer
{
    int in;
    int out;

    // cv::Mat W;
    // cv::Mat b;
    // cv::Mat z;
    // cv::Mat a;
    // cv::Mat delta;

    std::vector<float> W;
    std::vector<float> b;
    std::vector<float> z;
    std::vector<float> a;
    std::vector<float> delta;
};

// class HLayer
// {
//     private:
//     Layer layer;

//     public:
//     HLayer(int in, int out);
// };

class HiddenLayer
{
    private:
    int nodes;
    std::vector<cv::Mat> weight;
    cv::Mat bias;
    cv::Mat w_sum;
    cv::Mat activation;
    cv::Mat part_dL_db;
    std::vector<cv::Mat> dL_dB;
    std::vector<cv::Mat> dL_dW;


    public:
    HiddenLayer(int nodes, int inputSize);

    int getNodes();
    std::vector<cv::Mat> getWeight();
    cv::Mat getBias();
    void displayWeight();
    void displayBias();
    void setWeightedSum(float w);
    // void setActivation(int row, float z);
    void setActivation(float z);
    cv::Mat getWeightedSum();
    cv::Mat getActivation();
    
    void setUpdatedBias(int row, float bias_val);
    cv::Mat getUpdatedBias();
    
    void setTotalUpdatedBias(cv::Mat updatedBias);
    std::vector<cv::Mat> getTotalUpdatedBias();

    // void updateWeightVal(float weight_val);
};

class NeuNetModel
{
    private:
    cv::Mat inputLayer;
    std::vector<HiddenLayer> hiddenLayers;
    cv::Mat outputLayer;
    cv::Mat yHats;

    public:
    NeuNetModel(cv::Mat inputLayer, std::vector<HiddenLayer> hiddenLayers, cv::Mat outputLayers);


};