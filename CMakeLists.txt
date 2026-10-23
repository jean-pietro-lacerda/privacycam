cmake_minimum_required(VERSION 3.16)

project(PrivacyCam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc objdetect imgcodecs dnn videoio highgui)
find_package(Threads REQUIRED)
if(NOT OpenCV_VERSION VERSION_LESS 5)
    message(FATAL_ERROR "PrivacyCam requires OpenCV 4.x")
endif()

add_library(privacycam_core
    cpp/src/FaceDetector.cpp
    cpp/src/FaceAnonymizer.cpp
    cpp/src/VideoProcessor.cpp
)
target_include_directories(privacycam_core PUBLIC cpp/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(privacycam_core PUBLIC
    opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_imgcodecs
    opencv_dnn
    opencv_videoio
)

add_executable(privacycam cpp/src/main.cpp cpp/src/WebcamPipeline.cpp cpp/src/LatestFrameBuffer.cpp)
target_link_libraries(privacycam PRIVATE privacycam_core opencv_highgui Threads::Threads)
