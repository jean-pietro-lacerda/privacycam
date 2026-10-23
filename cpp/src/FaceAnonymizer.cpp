#include "FaceAnonymizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

using namespace std;

FaceAnonymizer::FaceAnonymizer(int pixelSize) : pixelSize_(pixelSize)
{
    if (pixelSize < 2)
        throw invalid_argument("Intensidade da pixelizacao deve ser um inteiro maior ou igual a 2.");
}

void FaceAnonymizer::anonymize(cv::Mat& image, const vector<FaceDetection>& faces) const
{
    if (image.empty() || image.type() != CV_8UC3)
        throw invalid_argument("Imagem invalida para anonimizacao: esperado BGR de 8 bits.");

    vector<cv::Rect> regions;
    regions.reserve(faces.size());
    for (const auto& face : faces)
    {
        const auto& bounds = face.bounds;
        if (!isfinite(bounds.x) || !isfinite(bounds.y) || !isfinite(bounds.width) ||
            !isfinite(bounds.height) || bounds.width <= 0 || bounds.height <= 0)
            throw invalid_argument("Coordenadas faciais invalidas.");

        const double marginX = bounds.width * 0.15;
        const double marginY = bounds.height * 0.15;
        const int left = static_cast<int>(floor(clamp(bounds.x - marginX, 0.0, double(image.cols))));
        const int top = static_cast<int>(floor(clamp(bounds.y - marginY, 0.0, double(image.rows))));
        const int right = static_cast<int>(ceil(clamp(double(bounds.x) + bounds.width + marginX, 0.0, double(image.cols))));
        const int bottom = static_cast<int>(ceil(clamp(double(bounds.y) + bounds.height + marginY, 0.0, double(image.rows))));
        if (right <= left || bottom <= top)
            throw invalid_argument("Regiao facial fora da imagem.");
        regions.emplace_back(left, top, right - left, bottom - top);
    }

    for (const auto& region : regions)
    {
        cv::Mat roi = image(region);
        cv::Mat reduced;
        const cv::Size size(max(1, roi.cols / pixelSize_), max(1, roi.rows / pixelSize_));
        cv::resize(roi, reduced, size, 0, 0, cv::INTER_AREA);
        cv::resize(reduced, roi, roi.size(), 0, 0, cv::INTER_NEAREST);
    }
}
