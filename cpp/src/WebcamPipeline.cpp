#include "WebcamPipeline.hpp"
#include "LatestFrameBuffer.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <exception>
#include <thread>
#include <utility>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace std;

namespace
{
    const string windowName = "PrivacyCam";

    struct WebcamWindow
    {
        ~WebcamWindow()
        {
            try
            {
                cv::destroyWindow(windowName);
            }
            catch (const cv::Exception&)
            {
            }
        }
    };
}

WebcamPipeline::WebcamPipeline(FaceDetector& detector, const FaceAnonymizer& anonymizer)
    : detector_(detector), anonymizer_(anonymizer)
{
}

void WebcamPipeline::run(int cameraIndex)
{
    if (cameraIndex < 0)
        throw invalid_argument("Indice de camera deve ser um inteiro nao negativo.");
    cv::VideoCapture camera(cameraIndex);
    if (!camera.isOpened())
        throw runtime_error("Camera indisponivel no ambiente atual, indice " + to_string(cameraIndex));
    display(camera);
}

void WebcamPipeline::run(const string& streamUrl)
{
    if (streamUrl.rfind("http://", 0) != 0)
        throw invalid_argument("Stream deve utilizar uma URL HTTP MJPEG local.");
    cv::VideoCapture camera;
    if (!camera.open(streamUrl, cv::CAP_FFMPEG, {
        cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000}))
        throw runtime_error("Falha ao conectar ao stream MJPEG.");
    display(camera);
}

void WebcamPipeline::display(cv::VideoCapture& camera)
{
#if defined(__linux__)
    const char* display = getenv("DISPLAY");
    const char* waylandDisplay = getenv("WAYLAND_DISPLAY");
    if ((!display || !*display) && (!waylandDisplay || !*waylandDisplay))
        throw runtime_error("Interface grafica indisponivel: DISPLAY e WAYLAND_DISPLAY ausentes.");
#endif

    cv::namedWindow(windowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    const WebcamWindow window;
    cv::resizeWindow(windowName, 960, 720);
    LatestFrameBuffer captured;
    LatestFrameBuffer processed;
    atomic<bool> stopping{false};
    mutex errorMutex;
    exception_ptr failure;
    const auto stop = [&]
    {
        stopping = true;
        captured.close();
        processed.close();
    };
    const auto fail = [&]
    {
        {
            lock_guard<mutex> lock(errorMutex);
            if (!failure)
                failure = current_exception();
        }
        stop();
    };
    thread captureThread;
    thread processingThread;
    const auto finish = [&]
    {
        stop();
        if (captureThread.joinable())
            captureThread.join();
        if (processingThread.joinable())
            processingThread.join();
    };
    try
    {
        captureThread = thread([&]
        {
            try
            {
                while (!stopping)
                {
                    cv::Mat frame;
                    if (!camera.read(frame) || frame.empty())
                        throw runtime_error("Falha de captura ou desconexao da camera/stream.");
                    if (!captured.push({move(frame), 0}))
                        break;
                }
            }
            catch (...)
            {
                fail();
            }
        });
        processingThread = thread([&]
        {
            try
            {
                BufferedFrame frame;
                while (captured.wait(frame))
                {
                    if (stopping)
                        break;
                    const auto faces = detector_.detect(frame.image);
                    anonymizer_.anonymize(frame.image, faces);
                    frame.faces = faces.size();
                    if (!processed.push(move(frame)))
                        break;
                }
            }
            catch (...)
            {
                fail();
            }
        });
        auto previousFrame = chrono::steady_clock::now();
        while (!stopping)
        {
            BufferedFrame frame;
            if (processed.waitFor(frame, chrono::milliseconds(10)) && !stopping)
            {
                const auto now = chrono::steady_clock::now();
                const double seconds = chrono::duration<double>(now - previousFrame).count();
                previousFrame = now;
                const double fps = seconds > 0 ? 1.0 / seconds : 0;
                ostringstream label;
                label << "Rostos: " << frame.faces << " | FPS: " << fixed << setprecision(1) << fps;
                cv::putText(frame.image, label.str(), cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
                cv::putText(frame.image, label.str(), cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                cv::imshow(windowName, frame.image);
            }
            const int key = cv::waitKey(1) & 0xff;
            if (key == 27 || key == 'q' || key == 'Q' ||
                cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1)
                break;
        }
    }
    catch (...)
    {
        finish();
        throw;
    }
    finish();
    if (failure)
        rethrow_exception(failure);
}
