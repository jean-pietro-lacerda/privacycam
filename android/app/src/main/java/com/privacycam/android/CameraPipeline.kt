package com.privacycam.android

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.camera2.CameraCharacteristics
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Size
import android.util.Log
import androidx.camera.core.Camera
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicLong

class CameraPipeline(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val worker: ExecutorService,
    private val initialize: () -> Unit,
    private val onFrame: (Bitmap, Int, Double, Long) -> Unit,
    private val onError: (String) -> Unit
) {
    private val handler = Handler(Looper.getMainLooper())
    private val generation = AtomicLong()
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var camera: Camera? = null
    private var initialized = false
    private val deliveryLock = Any()
    private var pendingFrame: ProcessedFrame? = null
    private var deliveryPosted = false
    private var profiling = false
    private var deliverySamples = 0L
    private var deliveryWaitNanos = 0L
    private var deliveryApplyNanos = 0L
    private var deliveryWindow = 0L

    private data class ProcessedFrame(
        val bitmap: Bitmap,
        val count: Int,
        val fps: Double,
        val session: Long,
        val completedAt: Long,
        val capturedAt: Long
    )

    private val deliverFrame = Runnable {
        val frame = synchronized(deliveryLock) {
            deliveryPosted = false
            pendingFrame.also { pendingFrame = null }
        }
        if (frame != null) {
            val applying = System.nanoTime()
            if (frame.session == generation.get() && applying - frame.completedAt <= 100_000_000L) {
                onFrame(frame.bitmap, frame.count, frame.fps, frame.capturedAt)
                if (profiling) {
                    ++deliverySamples
                    deliveryWaitNanos += applying - frame.completedAt
                    deliveryApplyNanos += System.nanoTime() - applying
                    if (applying - deliveryWindow >= 5_000_000_000L) {
                        Log.d("PrivacyCam", "UI avg ($deliverySamples frames): wait=${deliveryWaitNanos / deliverySamples / 1_000_000.0} ms, apply=${deliveryApplyNanos / deliverySamples / 1_000_000.0} ms")
                        deliverySamples = 0
                        deliveryWaitNanos = 0
                        deliveryApplyNanos = 0
                        deliveryWindow = applying
                    }
                }
            } else frame.bitmap.recycle()
        }
    }

    private fun publish(frame: ProcessedFrame) {
        synchronized(deliveryLock) {
            if (frame.session != generation.get()) {
                frame.bitmap.recycle()
                return
            }
            pendingFrame?.bitmap?.recycle()
            pendingFrame = frame
            if (!deliveryPosted) {
                deliveryPosted = true
                if (!handler.post(deliverFrame)) {
                    deliveryPosted = false
                    pendingFrame = null
                    frame.bitmap.recycle()
                }
            }
        }
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    fun start(lensFacing: Int, rotation: Int, inferenceMaxSide: Int = 0, profile: Boolean = true, confidence: Float = 0.6f) {
        stop()
        profiling = profile
        deliverySamples = 0
        deliveryWaitNanos = 0
        deliveryApplyNanos = 0
        deliveryWindow = System.nanoTime()
        val session = generation.get()
        try {
            val future = ProcessCameraProvider.getInstance(context.applicationContext)
            future.addListener({
                if (session != generation.get()) return@addListener
                try {
                    val available = future.get()
                    provider = available
                    val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                    check(available.hasCamera(selector)) { "Câmera selecionada indisponível." }
                    val info = selector.filter(available.availableCameraInfos).first()
                    val realtime = runCatching {
                        Camera2CameraInfo.from(info).getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
                    }.getOrDefault(false)
                    var timestampOffset: Long? = if (realtime) System.nanoTime() - SystemClock.elapsedRealtimeNanos() else null
                    if (!realtime) Log.w("PrivacyCam", "Relógio da câmera sem referência REALTIME; sincronia audiovisual usa estimativa inicial.")
                    val selectorResolution = ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(ResolutionStrategy(
                            Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        ))
                        .build()
                    val useCase = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setResolutionSelector(selectorResolution)
                        .setTargetRotation(rotation)
                        .build()
                    var started = 0L
                    var processed = 0L
                    var timingWindow = System.nanoTime()
                    var timingSamples = 0L
                    var zeroDetections = 0L
                    var totalDetections = 0L
                    val timingTotals = DoubleArray(4)
                    val timings = if (profile) DoubleArray(6) else null
                    var resolutionLogged = false
                    useCase.setAnalyzer(worker) { proxy ->
                        val width = proxy.width
                        val height = proxy.height
                        if (timestampOffset == null) timestampOffset = System.nanoTime() - proxy.imageInfo.timestamp
                        val capturedAt = proxy.imageInfo.timestamp + timestampOffset
                        analyze(proxy, session, lensFacing, inferenceMaxSide, timings, confidence, capturedAt) { now, count ->
                            if (!resolutionLogged) {
                                Log.i("PrivacyCam", "CameraX analysis: ${width}x${height}, RGBA_8888; confidence=$confidence; ${NativeBridge.status()}")
                                resolutionLogged = true
                            }
                            val completed = System.nanoTime()
                            if (timings != null) {
                                ++timingSamples
                                if (count == 0) ++zeroDetections
                                totalDetections += count
                                for (i in 0..3) timingTotals[i] += timings[i]
                                if (completed - timingWindow >= 5_000_000_000L) {
                                    Log.d("PrivacyCam", "Native avg ($timingSamples frames), CNN=${timings[4].toInt()}x${timings[5].toInt()}, confidence=$confidence, zero=$zeroDetections, faces=$totalDetections: prep=${timingTotals[0] / timingSamples} ms, YuNet=${timingTotals[1] / timingSamples} ms, anonymize=${timingTotals[2] / timingSamples} ms, output=${timingTotals[3] / timingSamples} ms")
                                    timingTotals.fill(0.0)
                                    timingSamples = 0
                                    zeroDetections = 0
                                    totalDetections = 0
                                    timingWindow = completed
                                }
                            }
                            if (started == 0L) started = now
                            ++processed
                            val elapsed = completed - started
                            processed * 1_000_000_000.0 / elapsed.coerceAtLeast(1L)
                        }
                    }
                    analysis = useCase
                    val bound = available.bindToLifecycle(owner, selector, useCase)
                    camera = bound
                    bound.cameraInfo.cameraState.observe(owner) { state ->
                        state.error?.let {
                            fail(session, "Falha na câmera (código ${it.code}).")
                        }
                    }
                } catch (error: Exception) {
                    fail(session, error.message ?: "Falha ao iniciar CameraX.")
                }
            }, ContextCompat.getMainExecutor(context))
        } catch (error: Exception) {
            fail(session, error.message ?: "Falha ao iniciar CameraX.")
        }
    }

    private fun analyze(
        proxy: ImageProxy,
        session: Long,
        lensFacing: Int,
        inferenceMaxSide: Int,
        timings: DoubleArray?,
        confidence: Float,
        capturedAt: Long,
        fps: (Long, Int) -> Double
    ) {
        var output: Bitmap? = null
        var transferred = false
        var proxyClosed = false
        try {
            if (session != generation.get()) return
            if (!initialized) {
                initialize()
                if (timings != null) Log.i("PrivacyCam", NativeBridge.status())
                initialized = true
                return
            }
            val started = System.nanoTime()
            check(proxy.format == PixelFormat.RGBA_8888 && proxy.planes.size == 1) {
                "CameraX não forneceu um frame RGBA compatível."
            }
            val rotation = proxy.imageInfo.rotationDegrees
            val swap = rotation == 90 || rotation == 270
            val bitmap = Bitmap.createBitmap(
                if (swap) proxy.height else proxy.width,
                if (swap) proxy.width else proxy.height,
                Bitmap.Config.ARGB_8888
            )
            output = bitmap
            val plane = proxy.planes[0]
            val nativeStarted = System.nanoTime()
            val count = NativeBridge.processFrame(
                plane.buffer.slice(), proxy.width, proxy.height,
                plane.rowStride, plane.pixelStride, rotation,
                lensFacing == CameraSelector.LENS_FACING_FRONT, bitmap, inferenceMaxSide, timings, confidence
            )
            check(count >= 0) { "Falha no processamento nativo da câmera." }
            proxy.close()
            proxyClosed = true
            if (session != generation.get()) return
            bitmap.setHasAlpha(false)
            if (timings != null) timings[0] += (nativeStarted - started) / 1_000_000.0
            val measuredFps = fps(started, count)
            if (session != generation.get()) return
            publish(ProcessedFrame(bitmap, count, measuredFps, session, System.nanoTime(), capturedAt))
            transferred = true
        } catch (error: Exception) {
            fail(session, error.message ?: "Falha no processamento da câmera.")
        } catch (error: LinkageError) {
            fail(session, error.message ?: "Falha JNI da câmera.")
        } finally {
            if (!transferred) {
                output?.recycle()
            }
            if (!proxyClosed) proxy.close()
        }
    }

    private fun fail(session: Long, message: String) {
        if (!generation.compareAndSet(session, session + 1)) return
        handler.post {
            if (session + 1 == generation.get()) {
                stop()
                onError(message)
            }
        }
    }

    fun stop() {
        generation.incrementAndGet()
        synchronized(deliveryLock) {
            pendingFrame?.bitmap?.recycle()
            pendingFrame = null
            handler.removeCallbacks(deliverFrame)
            deliveryPosted = false
        }
        analysis?.clearAnalyzer()
        camera?.cameraInfo?.cameraState?.removeObservers(owner)
        analysis?.let { provider?.unbind(it) }
        analysis = null
        camera = null
    }
}
