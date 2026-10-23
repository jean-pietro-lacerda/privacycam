#include "LatestFrameBuffer.hpp"

#include <utility>

using namespace std;

bool LatestFrameBuffer::push(BufferedFrame frame)
{
    {
        lock_guard<mutex> lock(mutex_);
        if (closed_)
            return false;
        pending_ = move(frame);
    }
    available_.notify_one();
    return true;
}

bool LatestFrameBuffer::wait(BufferedFrame& frame)
{
    unique_lock<mutex> lock(mutex_);
    available_.wait(lock, [this] { return closed_ || !pending_.image.empty(); });
    if (closed_)
        return false;
    frame = move(pending_);
    pending_ = {};
    return true;
}

bool LatestFrameBuffer::waitFor(BufferedFrame& frame, chrono::milliseconds timeout)
{
    unique_lock<mutex> lock(mutex_);
    if (!available_.wait_for(lock, timeout, [this] { return closed_ || !pending_.image.empty(); }) || closed_)
        return false;
    frame = move(pending_);
    pending_ = {};
    return true;
}

void LatestFrameBuffer::close()
{
    {
        lock_guard<mutex> lock(mutex_);
        closed_ = true;
        pending_ = {};
    }
    available_.notify_all();
}
