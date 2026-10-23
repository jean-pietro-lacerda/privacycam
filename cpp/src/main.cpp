#include "FaceDetector.hpp"
#include "FaceAnonymizer.hpp"
#include "VideoProcessor.hpp"
#include "WebcamPipeline.hpp"

#include <filesystem>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

int main(int argc, char* argv[])
{
    cout << "PrivacyCam\nOpenCV " << CV_VERSION << '\n';
    if (argc == 1)
        return 0;

    try
    {
        string imagePath;
        string videoPath;
        string streamUrl;
        string modelPath;
        string outputPath;
        int pixelSize = 24;
        float confidence = 0.9f;
        bool confidenceProvided = false;
        int cameraIndex = -1;
        bool cameraProvided = false;
        for (int index = 1; index < argc; index += 2)
        {
            const string option = argv[index];
            if (index + 1 >= argc)
                throw runtime_error("Valor ausente para " + option);
            if (option == "--camera")
            {
                if (cameraProvided)
                    throw runtime_error("Argumento duplicado: " + option);
                const string value = argv[index + 1];
                size_t parsed = 0;
                cameraIndex = stoi(value, &parsed);
                if (parsed != value.size() || cameraIndex < 0)
                    throw runtime_error("Indice de camera deve ser um inteiro nao negativo.");
                cameraProvided = true;
                continue;
            }
            if (option == "--confidence")
            {
                if (confidenceProvided)
                    throw runtime_error("Argumento duplicado: " + option);
                const string value = argv[index + 1];
                size_t parsed = 0;
                confidence = stof(value, &parsed);
                if (parsed != value.size() || !isfinite(confidence) || confidence <= 0 || confidence > 1)
                    throw runtime_error("Confianca deve ser um numero maior que 0 e menor ou igual a 1.");
                confidenceProvided = true;
                continue;
            }
            if (option == "--pixel-size")
            {
                const string value = argv[index + 1];
                size_t parsed = 0;
                pixelSize = stoi(value, &parsed);
                if (parsed != value.size())
                    throw runtime_error("Intensidade da pixelizacao invalida.");
                continue;
            }
            string* destination = nullptr;
            if (option == "--image") destination = &imagePath;
            else if (option == "--video") destination = &videoPath;
            else if (option == "--stream") destination = &streamUrl;
            else if (option == "--model") destination = &modelPath;
            else if (option == "--output") destination = &outputPath;
            else throw runtime_error("Argumento desconhecido: " + option);
            if (!destination->empty())
                throw runtime_error("Argumento duplicado: " + option);
            *destination = argv[index + 1];
        }
        const bool liveMode = cameraProvided || !streamUrl.empty();
        const int modeCount = int(!imagePath.empty()) + int(!videoPath.empty()) +
            int(cameraProvided) + int(!streamUrl.empty());
        if (modeCount != 1 || modelPath.empty() || (!liveMode && outputPath.empty()))
            throw runtime_error("Uso: privacycam (--image imagem | --video video | --camera indice | --stream URL) --model modelo.onnx [--output resultado (imagem/video)] [--pixel-size 24] [--confidence 0.9 (imagem/camera/stream)]");
        if (!videoPath.empty() && confidenceProvided)
            throw runtime_error("--confidence esta disponivel somente para imagens estaticas, camera e stream.");

        if (liveMode)
        {
            if (!outputPath.empty())
                throw runtime_error("O modo camera/stream nao grava arquivos e nao aceita --output.");
            const FaceAnonymizer anonymizer(pixelSize);
            FaceDetector detector(modelPath, confidence);
            WebcamPipeline pipeline(detector, anonymizer);
            if (cameraProvided)
                pipeline.run(cameraIndex);
            else
                pipeline.run(streamUrl);
            return 0;
        }

        const string& inputPath = imagePath.empty() ? videoPath : imagePath;
        if (filesystem::weakly_canonical(inputPath) == filesystem::weakly_canonical(outputPath) ||
            (filesystem::exists(inputPath) && filesystem::exists(outputPath) && filesystem::equivalent(inputPath, outputPath)))
            throw runtime_error("Entrada e saida devem ser arquivos diferentes.");

        const FaceAnonymizer anonymizer(pixelSize);

        if (!videoPath.empty())
        {
            FaceDetector detector(modelPath);
            VideoProcessor processor(detector, anonymizer);
            const auto result = processor.process(videoPath, outputPath);
            cout << "Frames processados: " << result.frames
                 << "\nFPS: " << result.fps
                 << "\nDuracao (s): " << result.frames / result.fps
                 << "\nDeteccoes faciais: " << result.detections
                 << "\nFrames com deteccoes: " << result.framesWithFaces << '\n';
            return 0;
        }

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty())
            throw runtime_error("Imagem inacessivel ou invalida: " + imagePath);

        FaceDetector detector(modelPath, confidence);
        const auto faces = detector.detect(image);
        anonymizer.anonymize(image, faces);
        if (!cv::imwrite(outputPath, image))
            throw runtime_error("Falha ao salvar imagem: " + outputPath);
        cout << "Rostos detectados: " << faces.size() << '\n';
        return 0;
    }
    catch (const exception& error)
    {
        cerr << "Erro: " << error.what() << '\n';
        return 1;
    }
}
