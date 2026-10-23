#pragma once

#include "FaceDetector.hpp"

class FaceAnonymizer
{
public:
    explicit FaceAnonymizer(int pixelSize = 24);
    void anonymize(cv::Mat& image, const std::vector<FaceDetection>& faces) const;

private:
    int pixelSize_;
};
