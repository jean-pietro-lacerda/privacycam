package com.privacycam.android

import android.graphics.Bitmap
import java.nio.ByteBuffer

object NativeBridge {
    init {
        System.loadLibrary("privacycam_jni")
    }

    external fun status(): String

    external fun initialize(modelPath: String)

    external fun process(image: Bitmap, confidence: Float = 0.6f): Int

    external fun processFrame(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        rotation: Int,
        mirror: Boolean,
        output: Bitmap,
        inferenceMaxSide: Int,
        timings: DoubleArray?,
        confidence: Float
    ): Int
}
