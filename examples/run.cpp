#include <iostream>
#include "tensor.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

#define w 400

struct DrawImage
{
    cv::Point start_point;
    cv::Point end_point;
    bool drawing = false;
    cv::Mat image;
};

void Draw(int event, int x, int y, int flags, void* img)
{
    DrawImage* d_img = static_cast<DrawImage*>(img);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        std::cout << "Drawing start" << std::endl;
        d_img->drawing = true;
        d_img->start_point = cv::Point(x,y);
        d_img->end_point = cv::Point(x,y);

    } else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (d_img->drawing)
        {
            d_img->end_point = cv::Point(x, y);
        }
    } else if (event == cv::EVENT_LBUTTONUP)
    {
        d_img->drawing = false;
        d_img->end_point = cv::Point(x,y);
        std::cout << "SP: " << d_img->start_point << ", EP: " << d_img->end_point << std::endl;
        cv::line(d_img->image, d_img->start_point, d_img->end_point, cv::Scalar(255,255,255,255), 5, cv::LINE_AA);
        cv::imshow("Iman Window", d_img->image);
    }
}

int main(int argc, char* argv[]) 
{
    std::cout << "Running Inference Engine" << std::endl;
    std::cout << "Testing .hpp file: " << add(1,5) << std::endl;
    int row = 3;
    int col = 3;
    std::vector<std::vector<float>> matA = createTensor(row, col);
    matA[0][0] = 1;
    matA[0][1] = 1;
    matA[0][2] = 1;

    matA[1][0] = 1;
    matA[1][1] = 1;
    matA[1][2] = 1;
       
    matA[2][0] = 1;
    matA[2][1] = 1;
    matA[2][2] = 1;
    // display2D(&matA);
    
    std::vector<std::vector<float>> matB = createTensor(row, col);
    matB[0][0] = 4;
    matB[1][0] = 2;
    matB[2][0] = 3;

    matB[0][1] = 5;
    matB[1][1] = 1;
    matB[2][1] = 2;

    // display2D(&matB);

    std::vector<std::vector<float>> mul = mulmat(matA, matB);
    // display2D(&mul);

    // // Open windows, draw and save
    // // 1. Create blank empty image and open it
    // cv::Mat black = cv::Mat(w, w, CV_8UC4);
    // cv::namedWindow("Iman Window", cv::WINDOW_FULLSCREEN);
    // cv::resizeWindow("Iman Window", 800, 800);
    // cv::imshow("Iman Window", black);

    // // Draw using mouse / trackpad
    // DrawImage d_img;
    // d_img.image = black;
    // cv::setMouseCallback("Iman Window", Draw, &d_img);
    // cv::waitKey(0);

    // // Save Image // Processing Image
    // cv::resize(d_img.image, d_img.image, cv::Size(28,28), 0, 1, cv::INTER_AREA);

    // We are going to feed into the hidden layer after this
    // cv::imwrite("/Users/imananuar/Documents/development/inference-engine/draw.jpg", d_img.image);
    cv::Mat input = cv::imread("/Users/imananuar/Documents/development/inference-engine/draw.jpg", cv::IMREAD_GRAYSCALE);
    int size = input.rows * input.cols;
    input.convertTo(input, CV_32F, 1.0/255.0);
    input = input.reshape(1, size);

    int N = 256;
    cv::Mat weight = cv::Mat::zeros(1, size, CV_32F);
    cv::Mat bias = cv::Mat::zeros(N, 1, CV_32F);
    cv::randu(weight, -0.1f, 0.1f);
    cv::randu(bias, -0.1f, 0.1f);

    activation_func(&input, &weight, &bias, N);
    
    cv::destroyWindow("Iman Window");

    // // read Image in opencv
    // cv::String imagePath = "/Users/imananuar/Documents/development/inference-engine/s_sparkling.jpg";
    // cv::Mat imageGs = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    // cv::Mat outImg;
    // imageGs.copyTo(outImg);
    // cv::resize(imageGs, outImg, cv::Size(w, w), 0, 1, cv::INTER_AREA);

    // cv::waitKey(0);
    cv::destroyAllWindows();
    std::cout << "All windows close. Exit Program. Thank you!\n" << std::endl;
    return 0;
}