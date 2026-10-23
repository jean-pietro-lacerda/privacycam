#include "FaceDetector.hpp"

#include <fstream>
#include <cmath>
#include <stdexcept>
#include <opencv2/dnn.hpp>

using namespace std;

FaceDetector::FaceDetector(const string& modelPath, float confidence)
{
    if (!isfinite(confidence) || confidence <= 0 || confidence > 1)
        throw invalid_argument("Confianca deve ser um numero maior que 0 e menor ou igual a 1.");
    if (!ifstream(modelPath, ios::binary).good())
        throw runtime_error("Modelo inacessivel: " + modelPath);

    try
    {
        detector_ = cv::FaceDetectorYN::create(modelPath, "", cv::Size(320, 320),
            confidence, 0.3f, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);
        if (detector_.empty())
            throw runtime_error("Falha ao carregar o modelo: " + modelPath);
    }
    catch (const cv::Exception& error)
    {
        throw runtime_error("Modelo invalido ou incompativel: " + string(error.what()));
    }
}

vector<FaceDetection> FaceDetector::detect(const cv::Mat& image)
{
    if (image.empty() || image.type() != CV_8UC3)
        throw runtime_error("Imagem invalida: esperado formato BGR de 8 bits.");

    cv::Mat faces;
    try
    {
        detector_->setInputSize(image.size());
        detector_->detect(image, faces);
    }
    catch (const cv::Exception& error)
    {
        throw runtime_error("Falha de inferencia ou modelo incompativel: " + string(error.what()));
    }

    if (!faces.empty() && (faces.type() != CV_32FC1 || faces.cols != 15))
        throw runtime_error("Resultado de inferencia incompativel.");

    vector<FaceDetection> detections;
    detections.reserve(faces.rows);
    for (int row = 0; row < faces.rows; ++row)
    {
        const float* face = faces.ptr<float>(row);
        detections.push_back({cv::Rect2f(face[0], face[1], face[2], face[3]), face[14]});
    }
    return detections;
}
