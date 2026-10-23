#include "FaceAnonymizer.hpp"
#include "FaceDetector.hpp"

#include <android/bitmap.h>
#include <jni.h>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace
{
unique_ptr<FaceDetector> detector;
mutex detectorMutex;
string detectorModelPath;
float detectorConfidence = 0.6f;

void configureConfidence(float confidence)
{
    if (confidence != 0.9f && confidence != 0.7f && confidence != 0.6f && confidence != 0.5f)
        throw invalid_argument("Limiar de confianca invalido.");
    if (confidence != detectorConfidence)
    {
        auto replacement = make_unique<FaceDetector>(detectorModelPath, confidence);
        detector = std::move(replacement);
        detectorConfidence = confidence;
    }
}

cv::Mat frameToBgr(const unsigned char* pixels, int64_t capacity, int width, int height,
    int rowStride, int pixelStride, int rotation)
{
    if (!pixels || width <= 0 || height <= 0 || pixelStride < 4 || rowStride <= 0)
        throw invalid_argument("Buffer RGBA invalido.");
    const int64_t rowBytes = int64_t(width - 1) * pixelStride + 4;
    const int64_t required = int64_t(height - 1) * rowStride + rowBytes;
    if (rowStride < rowBytes || capacity < required)
        throw invalid_argument("Strides ou tamanho do buffer RGBA invalidos.");
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270)
        throw invalid_argument("Rotacao de camera invalida.");

    thread_local cv::Mat packed;
    thread_local cv::Mat image;
    thread_local cv::Mat rotated;
    cv::Mat rgba;
    if (pixelStride == 4)
        rgba = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(pixels), rowStride);
    else
    {
        packed.create(height, width, CV_8UC4);
        rgba = packed;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                memcpy(rgba.ptr(y) + x * 4, pixels + int64_t(y) * rowStride + int64_t(x) * pixelStride, 4);
    }
    cv::cvtColor(rgba, image, cv::COLOR_RGBA2BGR);
    if (rotation == 90)
        cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
    else if (rotation == 180)
        cv::rotate(image, rotated, cv::ROTATE_180);
    else if (rotation == 270)
        cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    return rotation == 0 ? image : rotated;
}

cv::Mat prepareInference(const cv::Mat& image, int maxSide)
{
    if (image.empty() || (maxSide != 0 && maxSide != 480 && maxSide != 320))
        throw invalid_argument("Configuracao de inferencia invalida.");
    if (maxSide == 0 || max(image.cols, image.rows) <= maxSide)
        return image;
    const double scale = double(maxSide) / max(image.cols, image.rows);
    const cv::Size size(max(1, int(lround(image.cols * scale))), max(1, int(lround(image.rows * scale))));
    thread_local cv::Mat resized;
    cv::resize(image, resized, size, 0, 0, cv::INTER_AREA);
    return resized;
}

void restoreFaceCoordinates(vector<FaceDetection>& faces, const cv::Size& inference, const cv::Size& original)
{
    if (inference.width <= 0 || inference.height <= 0 || original.width <= 0 || original.height <= 0)
        throw invalid_argument("Dimensoes de inferencia invalidas.");
    if (inference == original)
        return;
    const float scaleX = float(original.width) / inference.width;
    const float scaleY = float(original.height) / inference.height;
    for (auto& face : faces)
    {
        face.bounds.x *= scaleX;
        face.bounds.y *= scaleY;
        face.bounds.width *= scaleX;
        face.bounds.height *= scaleY;
    }
}

void reportError(JNIEnv* env, const char* message)
{
    if (env->ExceptionCheck())
        return;
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type)
    {
        env->ThrowNew(type, message);
        env->DeleteLocalRef(type);
    }
}

class BitmapPixels
{
public:
    BitmapPixels(JNIEnv* env, jobject bitmap) : env_(env), bitmap_(bitmap)
    {
        if (AndroidBitmap_lockPixels(env_, bitmap_, &pixels_) != ANDROID_BITMAP_RESULT_SUCCESS)
            throw runtime_error("Falha ao acessar os pixels da imagem.");
    }

    ~BitmapPixels()
    {
        AndroidBitmap_unlockPixels(env_, bitmap_);
    }

    BitmapPixels(const BitmapPixels&) = delete;
    BitmapPixels& operator=(const BitmapPixels&) = delete;

    void* data() const { return pixels_; }

private:
    JNIEnv* env_;
    jobject bitmap_;
    void* pixels_ = nullptr;
};
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_privacycam_android_NativeBridge_status(JNIEnv* env, jobject)
{
    try
    {
        lock_guard<mutex> lock(detectorMutex);
        const string message = "Nucleo nativo carregado corretamente. OpenCV " CV_VERSION
            "; threads=" + to_string(cv::getNumThreads()) + "; CPUs=" + to_string(cv::getNumberOfCPUs())
            + "; confidence=" + to_string(detectorConfidence);
        return env->NewStringUTF(message.c_str());
    }
    catch (const exception& error)
    {
        reportError(env, error.what());
    }
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_privacycam_android_NativeBridge_initialize(JNIEnv* env, jobject, jstring modelPath)
{
    try
    {
        if (!modelPath)
            throw invalid_argument("Caminho do modelo ausente.");
        const char* path = env->GetStringUTFChars(modelPath, nullptr);
        if (!path)
            return;
        auto release = [env, modelPath](const char* value) { env->ReleaseStringUTFChars(modelPath, value); };
        unique_ptr<const char, decltype(release)> modelPathChars(path, release);
        lock_guard<mutex> lock(detectorMutex);
        if (!detector)
        {
            detectorModelPath = modelPathChars.get();
            detector = make_unique<FaceDetector>(detectorModelPath, detectorConfidence);
        }
    }
    catch (const exception& error)
    {
        reportError(env, error.what());
    }
    catch (...)
    {
        reportError(env, "Falha inesperada ao carregar o modelo.");
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_privacycam_android_NativeBridge_process(JNIEnv* env, jobject, jobject bitmap, jfloat confidence)
{
    try
    {
        lock_guard<mutex> lock(detectorMutex);
        if (!detector)
            throw runtime_error("Detector nao inicializado.");
        configureConfidence(confidence);
        AndroidBitmapInfo info{};
        if (!bitmap || AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
            info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width == 0 || info.height == 0)
            throw invalid_argument("Imagem invalida: esperado Bitmap ARGB_8888 de software.");

        cv::Mat image;
        {
            BitmapPixels pixels(env, bitmap);
            const cv::Mat rgba(info.height, info.width, CV_8UC4, pixels.data(), info.stride);
            cv::cvtColor(rgba, image, cv::COLOR_RGBA2BGR);
        }
        const auto faces = detector->detect(image);
        FaceAnonymizer anonymizer;
        anonymizer.anonymize(image, faces);
        {
            BitmapPixels pixels(env, bitmap);
            cv::Mat rgba(info.height, info.width, CV_8UC4, pixels.data(), info.stride);
            cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
        }
        return static_cast<jint>(faces.size());
    }
    catch (const exception& error)
    {
        reportError(env, error.what());
    }
    catch (...)
    {
        reportError(env, "Falha inesperada no processamento da imagem.");
    }
    return -1;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_privacycam_android_NativeBridge_processFrame(JNIEnv* env, jobject, jobject buffer,
    jint width, jint height, jint rowStride, jint pixelStride, jint rotation,
    jboolean mirror, jobject output, jint inferenceMaxSide, jdoubleArray timings, jfloat confidence)
{
    try
    {
        lock_guard<mutex> lock(detectorMutex);
        if (!detector)
            throw runtime_error("Detector nao inicializado.");
        if (!buffer || !output)
            throw invalid_argument("Frame ou imagem de saida ausente.");
        if (timings && env->GetArrayLength(timings) < 6)
            throw invalid_argument("Buffer de tempos invalido.");
        using Clock = chrono::steady_clock;
        const auto started = timings ? Clock::now() : Clock::time_point{};
        configureConfidence(confidence);
        const auto* pixels = static_cast<const unsigned char*>(env->GetDirectBufferAddress(buffer));
        const auto capacity = env->GetDirectBufferCapacity(buffer);
        cv::Mat image = frameToBgr(pixels, capacity, width, height, rowStride, pixelStride, rotation);
        AndroidBitmapInfo info{};
        if (AndroidBitmap_getInfo(env, output, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
            info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
            info.width != static_cast<uint32_t>(image.cols) || info.height != static_cast<uint32_t>(image.rows))
            throw invalid_argument("Bitmap de saida incompativel com o frame orientado.");

        const cv::Mat inference = prepareInference(image, inferenceMaxSide);
        const auto prepared = timings ? Clock::now() : Clock::time_point{};
        auto faces = detector->detect(inference);
        restoreFaceCoordinates(faces, inference.size(), image.size());
        const auto detected = timings ? Clock::now() : Clock::time_point{};
        FaceAnonymizer anonymizer;
        anonymizer.anonymize(image, faces);
        const auto anonymized = timings ? Clock::now() : Clock::time_point{};
        if (mirror)
            cv::flip(image, image, 1);
        {
            BitmapPixels locked(env, output);
            cv::Mat rgba(image.rows, image.cols, CV_8UC4, locked.data(), info.stride);
            cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
        }
        if (timings)
        {
            const auto completed = Clock::now();
            const double values[] = {
                chrono::duration<double, milli>(prepared - started).count(),
                chrono::duration<double, milli>(detected - prepared).count(),
                chrono::duration<double, milli>(anonymized - detected).count(),
                chrono::duration<double, milli>(completed - anonymized).count(),
                double(inference.cols), double(inference.rows)
            };
            env->SetDoubleArrayRegion(timings, 0, 6, values);
        }
        return static_cast<jint>(faces.size());
    }
    catch (const exception& error)
    {
        reportError(env, error.what());
    }
    catch (...)
    {
        reportError(env, "Falha inesperada no processamento do frame.");
    }
    return -1;
}
