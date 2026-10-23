cmake_minimum_required(VERSION 3.22.1)

project(PrivacyCamAndroid LANGUAGES CXX)

option(PRIVACYCAM_OPTIMIZE_DEBUG "Optimize native Debug code" ON)

find_package(OpenCV REQUIRED CONFIG)

set(PRIVACYCAM_CORE "${CMAKE_CURRENT_LIST_DIR}/../../../../../cpp")

add_library(privacycam_jni SHARED
    native_bridge.cpp
    "${PRIVACYCAM_CORE}/src/FaceDetector.cpp"
    "${PRIVACYCAM_CORE}/src/FaceAnonymizer.cpp"
)
target_include_directories(privacycam_jni PRIVATE "${PRIVACYCAM_CORE}/include")
if(PRIVACYCAM_OPTIMIZE_DEBUG)
    target_compile_options(privacycam_jni PRIVATE "$<$<CONFIG:Debug>:-O2>")
endif()
target_link_libraries(privacycam_jni PRIVATE OpenCV::opencv_java4 jnigraphics)
set_target_properties(privacycam_jni PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_link_options(privacycam_jni PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384"
)
