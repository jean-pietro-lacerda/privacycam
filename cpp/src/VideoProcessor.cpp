#include "VideoProcessor.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <opencv2/videoio.hpp>

using namespace std;

namespace
{
    struct TemporaryVideo
    {
        filesystem::path directory;
        ~TemporaryVideo()
        {
            error_code error;
            filesystem::remove_all(directory, error);
        }
    };
}

VideoProcessor::VideoProcessor(FaceDetector& detector, const FaceAnonymizer& anonymizer)
    : detector_(detector), anonymizer_(anonymizer)
{
}

VideoResult VideoProcessor::process(const string& inputPath, const string& outputPath)
{
    const filesystem::path output(outputPath);
    if (!filesystem::is_regular_file(inputPath))
        throw runtime_error("Video inexistente ou invalido: " + inputPath);
    if (filesystem::weakly_canonical(inputPath) == filesystem::weakly_canonical(output) ||
        (filesystem::exists(output) && filesystem::equivalent(inputPath, output)))
        throw runtime_error("Entrada e saida devem ser arquivos diferentes.");
    if (filesystem::exists(output))
        throw runtime_error("Arquivo de saida ja existe: " + outputPath);
    if (output.extension() != ".mp4")
        throw runtime_error("A saida de video deve utilizar a extensao .mp4.");

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened())
        throw runtime_error("Falha ao abrir video: " + inputPath);
    const double fps = capture.get(cv::CAP_PROP_FPS);
    const double width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    const double height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double expectedFrames = capture.get(cv::CAP_PROP_FRAME_COUNT);
    if (!isfinite(fps) || fps <= 0 || !isfinite(width) || !isfinite(height) ||
        width < 1 || height < 1 || width > numeric_limits<int>::max() || height > numeric_limits<int>::max() ||
        width != floor(width) || height != floor(height))
        throw runtime_error("FPS ou dimensoes invalidos.");
    const cv::Size size(static_cast<int>(width), static_cast<int>(height));

    const auto parent = output.has_parent_path() ? output.parent_path() : filesystem::path(".");
    filesystem::path directory;
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        const auto candidate = parent / (".privacycam-" +
            to_string(chrono::steady_clock::now().time_since_epoch().count()) + "-" + to_string(attempt));
        if (filesystem::create_directory(candidate))
        {
            directory = candidate;
            break;
        }
    }
    if (directory.empty())
        throw runtime_error("Falha ao criar arquivo temporario de video.");
    const TemporaryVideo temporary{directory};
    const auto temporaryPath = directory / "video.mp4";
    cv::VideoWriter writer;
    for (const int codec : {cv::VideoWriter::fourcc('m', 'p', '4', 'v'), cv::VideoWriter::fourcc('a', 'v', 'c', '1')})
    {
        if (writer.open(temporaryPath.string(), codec, fps, size))
            break;
    }
    if (!writer.isOpened())
        throw runtime_error("Falha ao abrir gravador MP4.");

    VideoResult result;
    result.fps = fps;
    cv::Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty() || frame.size() != size || frame.type() != CV_8UC3)
            throw runtime_error("Frame invalido durante leitura do video.");
        const auto faces = detector_.detect(frame);
        anonymizer_.anonymize(frame, faces);
        writer.write(frame);
        ++result.frames;
        result.detections += faces.size();
        if (!faces.empty())
            ++result.framesWithFaces;
    }
    if (result.frames == 0 || (isfinite(expectedFrames) && expectedFrames > 0 &&
        abs(double(result.frames) - expectedFrames) > 0.5))
        throw runtime_error("Leitura incompleta ou video sem frames.");
    capture.release();
    writer.release();

    cv::VideoCapture validation(temporaryPath.string());
    if (!validation.isOpened() || abs(validation.get(cv::CAP_PROP_FPS) - fps) > 0.001)
        throw runtime_error("Falha ao validar video gravado ou FPS divergente.");
    size_t writtenFrames = 0;
    while (validation.read(frame))
    {
        if (frame.empty() || frame.size() != size)
            throw runtime_error("Dimensoes divergentes no video gravado.");
        ++writtenFrames;
    }
    validation.release();
    if (writtenFrames != result.frames)
        throw runtime_error("Falha de gravacao: quantidade de frames divergente.");
    if (filesystem::exists(output))
        throw runtime_error("Arquivo de saida ja existe: " + outputPath);
    filesystem::rename(temporaryPath, output);
    return result;
}
