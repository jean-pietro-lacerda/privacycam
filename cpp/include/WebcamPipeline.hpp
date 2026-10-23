#pragma once

#include "FaceAnonymizer.hpp"
#include <opencv2/videoio.hpp>

class WebcamPipeline
{
public:
    WebcamPipeline(FaceDetector& detector, const FaceAnonymizer& anonymizer);
    void run(int cameraIndex);
    void run(const std::string& streamUrl);

private:
    FaceDetector& detector_;
    const FaceAnonymizer& anonymizer_;
    void display(cv::VideoCapture& camera);
};
