#include <iostream>
#include "tensor.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
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
    display2D(&matA);
    
    std::vector<std::vector<float>> matB = createTensor(row, col);
    matB[0][0] = 4;
    matB[1][0] = 2;
    matB[2][0] = 3;

    matB[0][1] = 5;
    matB[1][1] = 1;
    matB[2][1] = 2;

    display2D(&matB);

    std::vector<std::vector<float>> mul = mulmat(matA, matB);
    display2D(&mul);

    // read Image in opencv
    cv::String imagePath = "/Users/imananuar/Documents/development/inference-engine/s_sparkling.jpg";
    // cv::Mat imageColor = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat imageGs = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    // cv::Mat imageUc = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    cv::Mat outImg;
    imageGs.copyTo(outImg);

    
    // // Check if the images were loaded successfully
    // if (imageColor.empty()) {
        //     std::cout << "Error: ImageColor not found or unable to read." << std::endl;
        // } else {
            //     std::cout << "ImageColor loaded successfully!" << std::endl;
            // }
            // if (imageGs.empty()) {
                //     std::cout << "Error: ImageGs not found or unable to read." << std::endl;
                // } else {
                    //     std::cout << "ImageGs loaded successfully!" << std::endl;
                    // }
                    // if (imageUc.empty()) {
                        //     std::cout << "Error: ImageUc not found or unable to read." << std::endl;
                        // } else {
                            //     std::cout << "ImageUc loaded successfully!" << std::endl;
                            // };
    cv::resize(imageGs, outImg, cv::Size(28,28), 0, 0, cv::INTER_AREA);
    std::cout << "Image Matrix: \n" << outImg << std::endl;
                            

    cv::imshow("Displayed Output Image", outImg);
    cv::waitKey(0);
    cv::destroyAllWindows();
    std::cout << "All windows close. Exit Program. Thank you!\n" << std::endl;
    return 0;
}