#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>

struct FaceDetection
{
    cv::Rect2f bounds;
    float confidence;
};

class FaceDetector
{
public:
    explicit FaceDetector(const std::string& modelPath, float confidence = 0.9f);
    std::vector<FaceDetection> detect(const cv::Mat& image);

private:
    cv::Ptr<cv::FaceDetectorYN> detector_;
};
