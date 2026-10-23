import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.privacycam.android"
    compileSdk = 36
    buildToolsVersion = "36.0.0"
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.privacycam.android"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=c++_shared"
                arguments += "-DPRIVACYCAM_OPTIMIZE_DEBUG=${providers.gradleProperty("privacycamOptimizeDebug").orElse("ON").get()}"
            }
        }
    }

    buildFeatures {
        prefab = true
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
            pickFirsts += "**/libc++_shared.so"
        }
    }
}

dependencies {
    implementation("org.opencv:opencv:4.14.0")
    implementation("androidx.activity:activity:1.11.0")
    implementation("androidx.camera:camera-core:1.5.2")
    implementation("androidx.camera:camera-camera2:1.5.2")
    implementation("androidx.camera:camera-lifecycle:1.5.2")
}

val modelAssets = layout.buildDirectory.dir("generated/modelAssets")
val debugAssets = layout.buildDirectory.dir("generated/debugAssets")

val prepareModel by tasks.registering(Sync::class) {
    from(rootProject.file("../models/face_detection_yunet_2023mar.onnx"))
    into(modelAssets)
}

val prepareDebugImage by tasks.registering(Sync::class) {
    from(rootProject.file("../test_inputs/multiplos_rostos.jpeg")) {
        rename { "test_image.jpeg" }
    }
    into(debugAssets)
}

android.sourceSets.getByName("main").assets.srcDir(modelAssets)
android.sourceSets.getByName("debug").assets.srcDir(debugAssets)
tasks.named("preBuild").configure { dependsOn(prepareModel) }
tasks.matching { it.name == "preDebugBuild" }.configureEach { dependsOn(prepareDebugImage) }

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}
