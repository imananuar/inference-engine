#include "tensor.hpp"
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <customtype.hpp>
#include <random>

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
    std::vector<HiddenLayer> hiddenLayer_arr;
    float predict;
    cv::Mat raw_y;
    cv::Mat ce_loss;

    std::cout << inputMat->size() << std::endl;
    std::cout << "\nTRAINING MODEL" << std::endl;
    std::cout << "Input Matrix [col, row]: " << inputMat->at(0).size() << std::endl;
    std::cout << "Training Size: " << batchSize << std::endl;
    std::cout << "Layers: " << layers.size() << std::endl;
    std::cout << "Round Size: " << roundSize << std::endl;
    std::cout << "input answer = " << std::endl;
    std::cout << inputAns << std::endl;
    for (int round = 0; round < roundSize; round++)
    {
        // FORWARD PROPAGATION
        std::cout << "=====================================================================" << std::endl;
        std::cout << "\nTRAINING: ROUND = " << round;
        cv::Mat input = inputMat->at(round);
        std::cout << "..." << std::endl;
        for (int layer = 0; layer < layers.size(); layer++)
        {
            int inputSize = input.rows * input.cols;
            HiddenLayer HiddenLayer(layers[layer], inputSize);
            
            // std::cout << "Processing for layer: " << layer + 1;
            // std::cout << ", Node size:  " << layers[layer] << std::endl;
            float e_sum = 0;
            for (int node = 0; node < layers[layer]; node++)
            {
                // For each node, we will find x, y value for that node
                // x = z = ∑input*w + b
                float z;
                float a;
                for (int row = 0; row < input.rows; row++)
                {
                    // z = ∑input*w 
                    z += input.at<float>(row, 0) * HiddenLayer.getWeight().at(node).at<float>(0, row);
                }
                // z += b
                z += HiddenLayer.getBias().at<float>(node, 0);
                HiddenLayer.setWeightedSum(z);
                
                if ( layer != layers.size() -1 )
                {
                    // Set activation to leak-ReLU function if it is not the last layer
                    // HiddenLayer.setActivation(layer, std::max(0.01f, z));
                    HiddenLayer.setActivation(std::max(0.01f, z));
                } else
                // If last layer
                {
                    e_sum += std::exp(z);
                }
            }
            
            if ( layer == layers.size()-1 )
            {
                
                for (int i = 0; i < HiddenLayer.getWeightedSum().rows; i++)
                {
                    float y_hat = std::exp(HiddenLayer.getWeightedSum().at<float>(i, 0)) / e_sum;
                    // HiddenLayer.setActivation(i, y_hat);
                    HiddenLayer.setActivation(y_hat);
                    // y_hats.push_back(y_hat);
                    
                    float32_t ce = -1.0f * std::log(y_hat);
                    ce_loss.push_back(ce);
                }
                double maxVal;
                cv::Point maxLoc;
                cv::minMaxLoc(HiddenLayer.getActivation(), nullptr, &maxVal, nullptr, &maxLoc);
                std::cout << "\nSoftmax for round = " << round << std::endl;
                std::cout << HiddenLayer.getActivation() << std::endl;
                predict = maxLoc.y;
                raw_y = HiddenLayer.getWeightedSum();
            } 
            hiddenLayer_arr.push_back(HiddenLayer);
        }
    
        // std::cout << "\nTRAINING COMPLETED!\n" << std::endl;
        // std::cout << "Output Node (Before softmax function)" << std::endl;
        // std::cout << raw_y << std::endl;
        // std::cout << "\ny_hat (After apply softmax)" << std::endl;
        // std::cout << y_hats << std::endl;
        // std::cout << "\nCross Entropy Loss: " << std::endl;
        // std::cout << ce_loss << std::endl;
    
        int answer = inputAns.at<int32_t>(round, 0);
        std::cout << "\nObserved value = " << inputAns.at<int32_t>(round, 0) << std::endl;
        std::cout << "Predicted value = " << predict << std::endl;
        std::cout << "\n----------------------------------------------------------------" << std::endl;


        // BACK PROPAGATION
        std::cout << "\nRUN BACK PROPAGATION" << std::endl;
        for (int layer = layers.size() - 1; layer > 0; layer--)
        {
            HiddenLayer h_Layer = hiddenLayer_arr[layer * (round + 1) + round];
            
            std::cout << "\nHidden Layer for layer = " << layer  << std::endl;
            // std::cout << h_Layer.getUpdatedBias() << std::endl;
            std::cout << "\nactivation value = " << std::endl;
            std::cout << h_Layer.getActivation() << std::endl;

            for (int node = 0; node < hiddenLayer_arr[layer].getNodes(); node++)
            {
                // There are something wrong here. dL_dB is not correctly assigned
                float dL_dB = h_Layer.getActivation().at<float>(node, 0);
                // std::cout << dL_dB << " <- Before current" << std::endl;
                if ( node == answer )
                {
                    dL_dB -= 1.0f;
                }
                float current = h_Layer.getUpdatedBias().at<float>(node, 0);
                h_Layer.setUpdatedBias(node, current + dL_dB);
            }
            std::cout << "\nRound = " << round << std::endl;
            std::cout << "Layer = " << layer << std::endl;
            std::cout << "Updated_bias = " << std::endl;
            std::cout << h_Layer.getUpdatedBias() << std::endl;
            break;
        }

        if (round == 2)
        {
            break;
        }
    }
    
    std::cout << "\n" << std::endl;
}

void trainModelV2(int epoch, int training_size)
{
    int input = 3;
    int output = 5;
    std::vector<float> inputMat{0.23f, 0.47f, 0.9f};

    Layer hiddenLayer;
    hiddenLayer.in = input;
    hiddenLayer.out = output;

    std::vector<float> weight(input * output);
    std::random_device rd;
    std::mt19937 gen(rd());
    float stddev = std::sqrt( 2.0f / input );
    std::uniform_real_distribution<float> dist(-stddev, stddev);
    for (float &v : weight)
    {
        v = dist(gen);
    }

    hiddenLayer.W = weight;
    hiddenLayer.b = std::vector<float>(output, 0.01f);
    hiddenLayer.z = std::vector<float>(output, 0.0f);
    hiddenLayer.a = std::vector<float>(output, 0.0f);
    hiddenLayer.delta = std::vector<float>(output, 0.0f);


    Layer hiddenLayer2;
    int output2 = 4;
    hiddenLayer2.in = hiddenLayer.out;
    hiddenLayer2.out = output2;
    stddev = std::sqrt( 2.0f / hiddenLayer.out );
    std::uniform_real_distribution<float> dist2(-stddev, stddev);
    std::vector<float> weight2(hiddenLayer.out * output2);
    for (float &v : weight2)
    {
        v = dist2(gen);
    }
    hiddenLayer2.W = weight2;
    hiddenLayer2.b = std::vector<float>(output2, 0.01f);
    hiddenLayer2.z = std::vector<float>(output2, 0.0f);
    hiddenLayer2.a = std::vector<float>(output2, 0.0f);
    hiddenLayer2.delta = std::vector<float>(output2, 0.0f);

    // For each round
    int label = 1;
    float lr = 1.0f;
    for (int i = 0; i < epoch; i++)
    {
        for (int i = 0; i < output * output2; i++)
        {
            std::cout << "weight " << i << " = " << hiddenLayer2.W[i] << std::endl;
        }
        // First hidden Layer
        forward(hiddenLayer, inputMat);
        Relu(hiddenLayer);

        // Last hidden layer = Output Layer
        forward(hiddenLayer2, hiddenLayer.a);
        Softmax(hiddenLayer2);
        // Relu(hiddenLayer2);

        // forward(hiddenLayer3, hiddenLayer2.a);
        // Relu(hiddenLayer3);

        computeOutputDelta(hiddenLayer2, label);
        backward(hiddenLayer, hiddenLayer2);

        sgdUpdate(hiddenLayer2, hiddenLayer.a, lr);
        sgdUpdate(hiddenLayer, inputMat, lr);

        std::cout << "\n========================" << std::endl;
        for (int i = 0; i < output * output2; i++)
        {
            std::cout << "weight " << i << " = " << hiddenLayer2.W[i] << std::endl;
        }

        break;
    }
}

void sgdUpdate(Layer &L, std::vector<float> &input, float lr)
{
    for (int i = 0; i < L.out; i++)
    {
        L.b[i] -= L.delta[i] * lr;
        for (int j = 0; j < L.in; j++)
        {
            L.W[i * L.in + j] -= lr * L.delta[i] * input[i];
        }
    }
}

void backward(Layer &L, Layer &next)
{
    for (int i = 0; i < L.out; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < next.out; j++)
        {
            sum += next.delta[j] * next.W[(j * L.out) + i];
        }
        L.delta[i] = (L.a[i] > 0.01f ? sum : 0.0f);
    }
}


void computeOutputDelta(Layer &L, int label)
{
    for (int out = 0; out < L.out; out++)
    {
        L.delta[out] = L.a[out];
    }
    L.delta[label] -= 1.0f;

    // for (int out = 0; out < L.out; out++)
    // {
    //     std::cout << "y = " << L.a[out] << std::endl;
    //     std::cout << "delta = " << L.delta[out] << std::endl;
    // }
}

void forward(Layer &L, const std::vector<float> &input)
{
    for (int out = 0; out < L.out; out++)
    {
        for (int in = 0; in < L.in; in++)
        {
            // w = L.W[(L.in * out) + in];
            L.z[out] += input[in] * L.W[(L.in * out) + in];
        }
        L.z[out] += L.b[out];
    }
}

void Relu(Layer &L)
{
    for (int out = 0; out < L.out; out++)
    {
        L.a[out] = std::max(0.01f, L.z[out]);
    }
}

void Softmax(Layer &L)
{
    float e_sum;
    for (int out = 0; out < L.out; out++)
    {
        e_sum += std::exp(L.z[out]);
    }

    for (int out = 0; out < L.out; out++)
    {
        L.a[out] = std::exp(L.z[out]) / e_sum;
        // std::cout << "Softmax " << out << " = " << L.a[out] << std::endl;
    }
}