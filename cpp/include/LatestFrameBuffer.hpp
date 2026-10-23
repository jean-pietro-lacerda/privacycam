#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <opencv2/core.hpp>

struct BufferedFrame
{
    cv::Mat image;
    std::size_t faces = 0;
};

class LatestFrameBuffer
{
public:
    bool push(BufferedFrame frame);
    bool wait(BufferedFrame& frame);
    bool waitFor(BufferedFrame& frame, std::chrono::milliseconds timeout);
    void close();

private:
    std::mutex mutex_;
    std::condition_variable available_;
    BufferedFrame pending_;
    bool closed_ = false;
};
