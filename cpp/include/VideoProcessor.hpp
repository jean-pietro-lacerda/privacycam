#pragma once

#include "FaceAnonymizer.hpp"

#include <cstddef>
#include <string>

struct VideoResult
{
    std::size_t frames = 0;
    std::size_t detections = 0;
    std::size_t framesWithFaces = 0;
    double fps = 0;
};

class VideoProcessor
{
public:
    VideoProcessor(FaceDetector& detector, const FaceAnonymizer& anonymizer);
    VideoResult process(const std::string& inputPath, const std::string& outputPath);

private:
    FaceDetector& detector_;
    const FaceAnonymizer& anonymizer_;
};
