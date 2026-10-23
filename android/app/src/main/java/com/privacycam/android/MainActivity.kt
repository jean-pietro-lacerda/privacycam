package com.privacycam.android

import android.Manifest
import android.app.Dialog
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.display.DisplayManager
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Size
import android.view.Display
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.view.WindowInsets
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.lifecycle.Lifecycle
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

class MainActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val photoWorker = Executors.newSingleThreadExecutor()
    private val staticGeneration = AtomicLong()
    private lateinit var status: TextView
    private lateinit var galleryButton: ImageButton
    private var lastMediaUri: Uri? = null
    private var lastMediaType: String? = null
    private lateinit var metrics: TextView
    private lateinit var image: ImageView
    private lateinit var startButton: Button
    private lateinit var switchButton: Button
    private lateinit var inferenceButton: Button
    private lateinit var confidenceButton: Button
    private lateinit var photoButton: Button
    private lateinit var recordButton: Button
    private lateinit var recordStopButton: Button
    private lateinit var recordingStatus: TextView
    private lateinit var photoModeButton: Button
    private lateinit var videoModeButton: Button
    private var videoMode = false
    private var settingsDialog: Dialog? = null
    private lateinit var camera: CameraPipeline
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var cameraRequested = false
    private var pendingPermission = false
    private var resumeCamera = true
    private var permissionAttempted = false
    private var testing = false
    private var nativeReady = false
    private var cameraRotation = Surface.ROTATION_0
    private var lastMetricsUpdate = 0L
    private var lastFaceCount = -1
    private var inferenceMaxSide = 0
    private var confidence = 0.6f
    private var latestProcessedFrame: Bitmap? = null
    private var pendingPhotoFrame: Bitmap? = null
    private var savingPhoto = false
    private var recorder: VideoRecorder? = null
    private var pendingVideoStorage = false
    private var pendingMicrophone = false
    private var finishingVideo = false
    private var recordOrigin = 0L
    private var recordAudio = false
    private var previousOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
    private val recordingHandler = Handler(Looper.getMainLooper())
    private val recordingTicker = object : Runnable {
        override fun run() {
            if (recorder == null || finishingVideo || recordOrigin == 0L) return
            val seconds = (System.nanoTime() - recordOrigin) / 1_000_000_000L
            recordingStatus.text = getString(R.string.recording_status, seconds / 60, seconds % 60,
                getString(if (recordAudio) R.string.microphone_on else R.string.microphone_off))
            recordingHandler.postDelayed(this, 500)
        }
    }

    private val microphonePermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val requested = pendingMicrophone
        pendingMicrophone = false
        if (requested && !isDestroyed && !isFinishing && lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED))
            beginRecording(granted)
        if (!isDestroyed) updateControls()
    }

    private val storagePermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val videoRequested = pendingVideoStorage
        pendingVideoStorage = false
        val frame = pendingPhotoFrame
        pendingPhotoFrame = null
        if (!isDestroyed && !isFinishing) {
            if (videoRequested) {
                if (granted && lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) requestMicrophone()
                else if (!granted) Toast.makeText(this, R.string.video_storage_denied, Toast.LENGTH_LONG).show()
            } else if (granted && frame != null && lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED))
                savePhoto(frame)
            else if (!granted)
                Toast.makeText(this, R.string.storage_permission_denied, Toast.LENGTH_LONG).show()
            updateControls()
        }
    }

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) {
            val display = image.display ?: return
            if (cameraRequested && display.displayId == displayId && display.rotation != cameraRotation) {
                if (videoBusy()) stopCamera() else startCamera()
            }
        }
    }

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val requested = pendingPermission
        pendingPermission = false
        if (requested && !isFinishing && !isDestroyed) {
            if (granted && lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) startCamera()
            else if (!granted) {
                status.text = getString(R.string.permission_denied)
                updateControls()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        lensFacing = savedInstanceState?.getInt("lensFacing")
            ?: getPreferences(MODE_PRIVATE).getInt("lensFacing", CameraSelector.LENS_FACING_BACK)
        inferenceMaxSide = savedInstanceState?.getInt("inferenceMaxSide") ?: 0
        confidence = savedInstanceState?.getFloat("confidence") ?: 0.6f
        resumeCamera = true
        permissionAttempted = savedInstanceState?.getBoolean("permissionAttempted") ?: false

        videoMode = savedInstanceState?.getBoolean("videoMode") ?: false
        val title = TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 22f
            setTextColor(getColor(R.color.pc_text))
            setCompoundDrawablesRelativeWithIntrinsicBounds(R.drawable.privacycam_header_logo, 0, 0, 0)
            compoundDrawablePadding = dp(10)
            gravity = Gravity.CENTER_VERTICAL
        }
        status = TextView(this).apply {
            text = getString(R.string.ready)
            textSize = 13f
            setTextColor(getColor(R.color.pc_muted))
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(6), dp(12), dp(6))
        }
        metrics = TextView(this).apply {
            textSize = 13f
            setTextColor(getColor(R.color.pc_muted))
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(4), dp(8), dp(4))
            setBackgroundColor(0xB30B0F14.toInt())
        }
        image = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            contentDescription = getString(R.string.processed_image)
        }
        startButton = Button(this).apply {
            text = getString(R.string.retry_camera)
            setOnClickListener { startCamera() }
        }
        switchButton = Button(this).apply {
            text = getString(R.string.switch_camera)
            setOnClickListener {
                lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                    CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
                getPreferences(MODE_PRIVATE).edit().putInt("lensFacing", lensFacing).apply()
                if (cameraRequested) startCamera() else status.text = cameraName()
            }
        }
        inferenceButton = Button(this).apply {
            text = inferenceLabel()
            setOnClickListener {
                inferenceMaxSide = when (inferenceMaxSide) {
                    0 -> 480
                    480 -> 320
                    else -> 0
                }
                text = inferenceLabel()
                if (cameraRequested) startCamera()
            }
        }
        confidenceButton = Button(this).apply {
            text = getString(R.string.confidence, confidence)
            setOnClickListener {
                confidence = when (confidence) {
                    0.9f -> 0.7f
                    0.7f -> 0.6f
                    0.6f -> 0.5f
                    else -> 0.9f
                }
                text = getString(R.string.confidence, confidence)
                if (cameraRequested) startCamera()
            }
        }
        photoButton = Button(this).apply {
            text = getString(R.string.take_photo)
            setOnClickListener { takePhoto() }
        }
        recordButton = Button(this).apply {
            text = getString(R.string.start_recording)
            setOnClickListener { startRecording() }
        }
        recordStopButton = Button(this).apply {
            text = getString(R.string.stop_recording)
            setOnClickListener { stopRecording() }
        }
        recordingStatus = TextView(this).apply {
            gravity = Gravity.CENTER
            textSize = 14f
            setTextColor(getColor(R.color.pc_record))
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        val settings = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }
        listOf(confidenceButton, inferenceButton).forEach {
            styleControl(it)
            settings.addView(it, controlParams())
        }
        val panel = Dialog(this).apply {
            setTitle(R.string.settings)
            setContentView(ScrollView(this@MainActivity).apply { addView(settings) })
        }
        settingsDialog = panel
        val settingsButton = Button(this).apply {
            contentDescription = getString(R.string.settings)
            styleControl(this)
            foreground = getDrawable(R.drawable.ic_settings)
            foregroundGravity = Gravity.CENTER
            setOnClickListener {
                panel.show()
                panel.window?.setLayout(
                    minOf(resources.displayMetrics.widthPixels - dp(32), dp(420)),
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }
        }
        val closeSettings = Button(this).apply {
            text = getString(R.string.close_settings)
            styleControl(this)
            setOnClickListener { panel.dismiss() }
        }
        settings.addView(closeSettings, controlParams())
        val header = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            addView(title, LinearLayout.LayoutParams(0, dp(56), 1f))
            addView(settingsButton, LinearLayout.LayoutParams(dp(48), dp(48)))
        }
        val viewer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(getColor(R.color.pc_background))
            addView(FrameLayout(this@MainActivity).apply {
                setBackgroundColor(android.graphics.Color.BLACK)
                addView(image, FrameLayout.LayoutParams(-1, -1))
                addView(metrics, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.END).apply {
                    setMargins(dp(8), dp(8), dp(8), dp(8))
                })
            }, LinearLayout.LayoutParams(-1, 0, 1f))
            addView(status, LinearLayout.LayoutParams(-1, -2))
        }
        photoModeButton = Button(this).apply {
            text = getString(R.string.photo_mode)
            styleControl(this)
            setOnClickListener { videoMode = false; updateCapturePresentation() }
        }
        videoModeButton = Button(this).apply {
            text = getString(R.string.video_mode)
            styleControl(this)
            setOnClickListener { videoMode = true; updateCapturePresentation() }
        }
        val modes = LinearLayout(this).apply {
            addView(photoModeButton, LinearLayout.LayoutParams(0, dp(48), 1f))
            addView(videoModeButton, LinearLayout.LayoutParams(0, dp(48), 1f))
        }
        listOf(photoButton, recordButton, recordStopButton).forEach {
            it.background = getDrawable(R.drawable.capture_background)
            it.backgroundTintList = null
            it.setTextColor(getColor(R.color.pc_background))
            it.contentDescription = it.text
            it.text = ""
            it.setPadding(0, 0, 0, 0)
            it.minWidth = 0
            it.minimumWidth = 0
        }
        recordButton.isSelected = true
        recordStopButton.isActivated = true
        styleControl(switchButton)
        switchButton.contentDescription = getString(R.string.switch_camera)
        switchButton.text = ""
        switchButton.foreground = getDrawable(R.drawable.ic_switch)
        switchButton.foregroundGravity = Gravity.CENTER
        styleControl(startButton)
        val shutter = FrameLayout(this).apply {
            listOf(photoButton, recordButton, recordStopButton).forEach {
                addView(it, FrameLayout.LayoutParams(dp(76), dp(76), Gravity.CENTER))
            }
        }
        galleryButton = ImageButton(this).apply {
            contentDescription = getString(R.string.open_last_media)
            background = getDrawable(R.drawable.control_background)
            backgroundTintList = null
            scaleType = ImageView.ScaleType.CENTER_CROP
            setPadding(dp(4), dp(4), dp(4), dp(4))
            setImageResource(R.drawable.ic_gallery)
            setOnClickListener { openLastMedia() }
        }
        val actions = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            addView(galleryButton, LinearLayout.LayoutParams(dp(48), dp(48)))
            addView(shutter, LinearLayout.LayoutParams(0, dp(92), 1f))
            addView(switchButton, LinearLayout.LayoutParams(dp(48), dp(48)))
        }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(4), dp(12), dp(4))
            addView(modes)
            addView(actions)
            addView(startButton, controlParams())
        }
        val landscape = resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        val body = LinearLayout(this).apply {
            orientation = if (landscape) LinearLayout.HORIZONTAL else LinearLayout.VERTICAL
            addView(viewer, if (landscape) LinearLayout.LayoutParams(0, -1, 1f)
                else LinearLayout.LayoutParams(-1, 0, 1f))
            addView(ScrollView(this@MainActivity).apply {
                isFillViewport = true
                addView(controls)
            }, if (landscape) LinearLayout.LayoutParams(dp(200), -1)
                else LinearLayout.LayoutParams(-1, -2))
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setOnApplyWindowInsetsListener { view, insets ->
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                    view.setPadding(bars.left + dp(4), bars.top, bars.right + dp(4), bars.bottom + dp(4))
                } else {
                    view.setPadding(insets.systemWindowInsetLeft + dp(4), insets.systemWindowInsetTop,
                        insets.systemWindowInsetRight + dp(4), insets.systemWindowInsetBottom + dp(4))
                }
                insets
            }
            setBackgroundColor(getColor(R.color.pc_background))
            setPadding(dp(4), 0, dp(4), dp(4))
            addView(header)
            addView(recordingStatus)
            addView(body, LinearLayout.LayoutParams(-1, 0, 1f))
        })
        camera = CameraPipeline(this, this, worker, ::initializeNative, { bitmap, count, fps, capturedNs ->
            if (cameraRequested && lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
                image.setImageBitmap(bitmap)
                latestProcessedFrame = bitmap
                recorder?.offer(bitmap, capturedNs)
                photoButton.isEnabled = !savingPhoto && pendingPhotoFrame == null
                recordButton.isEnabled = !videoBusy() && pendingPhotoFrame == null
                val now = System.nanoTime()
                if (count != lastFaceCount || now - lastMetricsUpdate >= 250_000_000L) {
                    metrics.text = getString(R.string.camera_metrics, fps, count)
                    lastMetricsUpdate = now
                }
                if (lastFaceCount < 0 || (count == 0) != (lastFaceCount == 0))
                    status.text = if (count == 0) getString(R.string.zero_faces) else cameraName()
                lastFaceCount = count
            } else bitmap.recycle()
        }, { message ->
            stopRecording()
            cameraRequested = false
            latestProcessedFrame = null
            pendingPhotoFrame = null
            image.setImageDrawable(null)
            metrics.text = ""
            status.text = getString(R.string.failure, message)
            updateControls()
        })
        updateControls()
        val savedMedia = savedInstanceState?.getString("lastMediaUri")
        val savedType = savedInstanceState?.getString("lastMediaType")
        if (savedMedia != null && savedType != null) showSavedMedia(Uri.parse(savedMedia), savedType)
    }

    private fun openLastMedia() {
        val uri = lastMediaUri
        if (uri == null) {
            Toast.makeText(this, R.string.no_saved_media, Toast.LENGTH_SHORT).show()
            return
        }
        try {
            startActivity(Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, lastMediaType)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            })
        } catch (error: ActivityNotFoundException) {
            Toast.makeText(this, R.string.media_open_failed, Toast.LENGTH_LONG).show()
        } catch (error: SecurityException) {
            Toast.makeText(this, R.string.media_open_failed, Toast.LENGTH_LONG).show()
        }
    }

    private fun showSavedMedia(uri: Uri, type: String) {
        lastMediaUri = uri
        lastMediaType = type
        galleryButton.setImageResource(R.drawable.ic_gallery)
        if (photoWorker.isShutdown) return
        photoWorker.execute {
            val thumbnail = try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    contentResolver.loadThumbnail(uri, Size(dp(48), dp(48)), null)
                } else if (type.startsWith("video/")) {
                    val retriever = MediaMetadataRetriever()
                    try {
                        retriever.setDataSource(this, uri)
                        val frame = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                            retriever.getScaledFrameAtTime(
                                -1, MediaMetadataRetriever.OPTION_CLOSEST_SYNC, dp(48), dp(48)
                            )
                        } else retriever.getFrameAtTime(-1)
                        frame?.let {
                            val scale = dp(48).toFloat() / maxOf(it.width, it.height)
                            val scaled = Bitmap.createScaledBitmap(
                                it, maxOf(1, (it.width * scale).toInt()),
                                maxOf(1, (it.height * scale).toInt()), true
                            )
                            if (scaled !== it) it.recycle()
                            scaled
                        }
                    } finally {
                        retriever.release()
                    }
                } else {
                    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
                    val options = BitmapFactory.Options().apply {
                        inSampleSize = 1
                        while (bounds.outWidth / inSampleSize > dp(96) &&
                            bounds.outHeight / inSampleSize > dp(96)) {
                            inSampleSize *= 2
                        }
                    }
                    contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
                }
            } catch (error: Exception) {
                null
            } catch (error: OutOfMemoryError) {
                null
            }
            runOnUiThread {
                if (!isDestroyed && !isFinishing && lastMediaUri == uri) {
                    if (thumbnail != null) galleryButton.setImageBitmap(thumbnail)
                } else thumbnail?.recycle()
            }
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun controlParams() = LinearLayout.LayoutParams(-1, -2).apply {
        setMargins(dp(3), dp(3), dp(3), dp(3))
    }

    private fun styleControl(button: Button, icon: Int = 0) {
        button.background = getDrawable(R.drawable.control_background)
        button.backgroundTintList = null
        button.setTextColor(ColorStateList(
            arrayOf(intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
            intArrayOf(getColor(R.color.pc_muted), getColor(R.color.pc_text))
        ))
        button.isAllCaps = false
        button.textSize = 14f
        button.minHeight = dp(48)
        button.minimumHeight = dp(48)
        button.setPadding(dp(12), dp(8), dp(12), dp(8))
        if (icon != 0) {
            button.setCompoundDrawablesRelativeWithIntrinsicBounds(icon, 0, 0, 0)
            button.compoundDrawablePadding = dp(8)
        }
    }

    private fun updateCapturePresentation() {
        val recording = videoBusy()
        photoModeButton.isSelected = !videoMode
        videoModeButton.isSelected = videoMode
        photoModeButton.isEnabled = !recording
        videoModeButton.isEnabled = !recording
        photoButton.visibility = if (!videoMode && !recording) View.VISIBLE else View.GONE
        recordButton.visibility = if (videoMode && !recording) View.VISIBLE else View.GONE
        recordStopButton.visibility = if (recording) View.VISIBLE else View.GONE
        startButton.visibility = if (!cameraRequested) View.VISIBLE else View.GONE
    }

    private fun cameraName(): String = getString(
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) R.string.front_camera else R.string.back_camera
    )

    private fun inferenceLabel(): String = getString(when (inferenceMaxSide) {
        480 -> R.string.inference_480
        320 -> R.string.inference_320
        else -> R.string.inference_original
    })

    private fun updateControls() {
        galleryButton.isEnabled = !videoBusy()
        startButton.isEnabled = !cameraRequested && !testing && !videoBusy()
        switchButton.isEnabled = !testing && !videoBusy()
        inferenceButton.isEnabled = !testing && !videoBusy()
        confidenceButton.isEnabled = !testing && !videoBusy()
        photoButton.isEnabled = cameraRequested && latestProcessedFrame != null && !savingPhoto && pendingPhotoFrame == null
        photoButton.contentDescription = getString(if (savingPhoto) R.string.saving_photo else R.string.take_photo)
        recordButton.isEnabled = cameraRequested && latestProcessedFrame != null && !videoBusy() && pendingPhotoFrame == null
        recordStopButton.isEnabled = videoBusy() && !finishingVideo
        updateCapturePresentation()
    }

    private fun startCamera() {
        if (testing || videoBusy() || isDestroyed || isFinishing) return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            if (!pendingPermission) {
                pendingPermission = true
                permissionAttempted = true
                permission.launch(Manifest.permission.CAMERA)
            }
            return
        }
        staticGeneration.incrementAndGet()
        cameraRequested = true
        latestProcessedFrame = null
        pendingPhotoFrame = null
        lastFaceCount = -1
        lastMetricsUpdate = 0L
        image.setImageDrawable(null)
        metrics.text = ""
        status.text = getString(R.string.camera_starting)
        updateControls()
        cameraRotation = (
            image.display?.rotation
                ?: getSystemService(DisplayManager::class.java).getDisplay(Display.DEFAULT_DISPLAY)?.rotation
                ?: Surface.ROTATION_0
        )
        camera.start(lensFacing, cameraRotation, inferenceMaxSide, intent.getBooleanExtra("profile", true), confidence)
    }

    private fun stopCamera() {
        stopRecording()
        pendingPermission = false
        cameraRequested = false
        latestProcessedFrame = null
        pendingPhotoFrame = null
        camera.stop()
        image.setImageDrawable(null)
        metrics.text = ""
        status.text = getString(R.string.camera_stopped)
        updateControls()
    }

    private fun initializeNative() {
        if (nativeReady) return
        val model = File(filesDir, "face_detection_yunet_2023mar.onnx")
        if (!model.isFile) {
            val temporary = File.createTempFile("yunet-", ".tmp", filesDir)
            try {
                assets.open(model.name).use { source ->
                    temporary.outputStream().use { target -> source.copyTo(target) }
                }
                check(temporary.renameTo(model)) { "Falha ao preparar o modelo." }
            } finally {
                temporary.delete()
            }
        }
        NativeBridge.initialize(model.absolutePath)
        nativeReady = true
    }

    private fun takePhoto() {
        if (savingPhoto || pendingPhotoFrame != null || pendingVideoStorage) return
        val frame = latestProcessedFrame
        if (!cameraRequested || frame == null || frame.isRecycled) {
            Toast.makeText(this, R.string.photo_unavailable, Toast.LENGTH_SHORT).show()
            return
        }
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            pendingPhotoFrame = frame
            updateControls()
            storagePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            return
        }
        savePhoto(frame)
    }

    private fun savePhoto(frame: Bitmap) {
        if (savingPhoto || isDestroyed || photoWorker.isShutdown) return
        savingPhoto = true
        updateControls()
        val app = applicationContext
        photoWorker.execute {
            var snapshot: Bitmap? = null
            var message: String
            var savedUri: Uri? = null
            try {
                val stable = frame.copy(Bitmap.Config.ARGB_8888, false)
                    ?: error("Falha ao copiar o frame processado.")
                snapshot = stable
                savedUri = PhotoStore(app).save(stable)
                message = getString(R.string.photo_saved, stable.width, stable.height)
            } catch (error: Exception) {
                message = getString(R.string.photo_failed, error.message ?: error.javaClass.simpleName)
            } catch (error: OutOfMemoryError) {
                message = getString(R.string.photo_failed, "Memória insuficiente.")
            } finally {
                snapshot?.recycle()
            }
            runOnUiThread {
                savingPhoto = false
                if (!isDestroyed && !isFinishing) {
                    updateControls()
                    savedUri?.let { showSavedMedia(it, "image/jpeg") }
                }
                Toast.makeText(app, message, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun videoBusy(): Boolean = recorder != null || pendingVideoStorage || pendingMicrophone

    private fun startRecording() {
        if (videoBusy() || !cameraRequested || latestProcessedFrame == null || pendingPhotoFrame != null) return
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            pendingVideoStorage = true
            updateControls()
            storagePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        } else requestMicrophone()
    }

    private fun requestMicrophone() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) beginRecording(true)
        else {
            pendingMicrophone = true
            updateControls()
            microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun beginRecording(audio: Boolean) {
        val frame = latestProcessedFrame ?: return
        if (!cameraRequested || recorder != null || isDestroyed || isFinishing) return
        previousOrientation = requestedOrientation
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LOCKED
        finishingVideo = false
        recordOrigin = 0
        recordingStatus.setTextColor(getColor(R.color.pc_muted))
        recordingStatus.text = getString(R.string.recording_starting)
        lateinit var session: VideoRecorder
        session = VideoRecorder(applicationContext, frame.width, frame.height, audio, { hasAudio, origin ->
            if (recorder === session && !finishingVideo && !isDestroyed) {
                recordAudio = hasAudio
                recordOrigin = origin
                recordingStatus.setTextColor(getColor(R.color.pc_record))
                recordingHandler.post(recordingTicker)
                updateControls()
            }
        }, { uri, error ->
            if (recorder === session) {
                recorder = null
                finishingVideo = false
                recordOrigin = 0
                recordingHandler.removeCallbacks(recordingTicker)
                if (!isDestroyed) {
                    requestedOrientation = previousOrientation
                    recordingStatus.text = ""
                    updateControls()
                    if (
                        resumeCamera && !cameraRequested &&
                        lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED) &&
                        checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                    ) {
                        resumeCamera = false
                        startCamera()
                    }
                }
            }
            if (uri != null && !isDestroyed && !isFinishing) showSavedMedia(uri, "video/mp4")
            val message = if (uri != null) getString(R.string.video_saved)
                else getString(R.string.video_failed, error ?: "Gravação incompleta.")
            Toast.makeText(applicationContext, message, Toast.LENGTH_LONG).show()
        })
        recorder = session
        updateControls()
        session.start()
    }

    private fun stopRecording() {
        pendingVideoStorage = false
        pendingMicrophone = false
        recorder?.let {
            finishingVideo = true
            it.stop()
            recordingHandler.removeCallbacks(recordingTicker)
            recordingStatus.setTextColor(getColor(R.color.pc_muted))
            recordingStatus.text = getString(R.string.recording_finishing)
        }
        if (::recordButton.isInitialized) updateControls()
    }

    private fun testImage() {
        if (testing || cameraRequested) return
        val session = staticGeneration.incrementAndGet()
        val testConfidence = confidence
        testing = true
        latestProcessedFrame = null
        pendingPhotoFrame = null
        updateControls()
        image.setImageDrawable(null)
        metrics.text = ""
        status.text = getString(R.string.processing)
        worker.execute {
            var bitmap: Bitmap? = null
            try {
                if (session != staticGeneration.get()) return@execute
                initializeNative()
                if (session != staticGeneration.get()) return@execute
                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inMutable = true
                    inScaled = false
                }
                val result = assets.open("test_image.jpeg").use {
                    BitmapFactory.decodeStream(it, null, options)
                } ?: error("Imagem de teste ausente ou inválida.")
                bitmap = result
                val count = NativeBridge.process(result, testConfidence)
                check(count >= 0) { "Falha no processamento nativo." }
                runOnUiThread {
                    if (session != staticGeneration.get() || isDestroyed || isFinishing) result.recycle()
                    else {
                        image.setImageBitmap(result)
                        status.text = getString(R.string.success, count)
                        testing = false
                        updateControls()
                    }
                }
            } catch (error: Exception) {
                bitmap?.recycle()
                showStaticError(session, error.message ?: error.javaClass.simpleName)
            } catch (error: LinkageError) {
                bitmap?.recycle()
                showStaticError(session, error.message ?: error.javaClass.simpleName)
            }
        }
    }

    private fun showStaticError(session: Long, message: String) {
        runOnUiThread {
            if (session == staticGeneration.get() && !isDestroyed && !isFinishing) {
                image.setImageDrawable(null)
                status.text = getString(R.string.failure, message)
                testing = false
                updateControls()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        val granted = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (resumeCamera && !cameraRequested && !pendingPermission && !videoBusy() &&
            (granted || !permissionAttempted)) {
            resumeCamera = false
            startCamera()
        } else if (!granted && permissionAttempted && !pendingPermission) {
            status.text = getString(R.string.permission_denied)
            updateControls()
        }
    }

    override fun onStart() {
        super.onStart()
        getSystemService(DisplayManager::class.java).registerDisplayListener(displayListener, null)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putInt("lensFacing", lensFacing)
        outState.putInt("inferenceMaxSide", inferenceMaxSide)
        outState.putFloat("confidence", confidence)
        outState.putBoolean("cameraRequested", cameraRequested || resumeCamera)
        outState.putBoolean("videoMode", videoMode)
        outState.putBoolean("permissionAttempted", permissionAttempted)
        outState.putString("lastMediaUri", lastMediaUri?.toString())
        outState.putString("lastMediaType", lastMediaType)
        super.onSaveInstanceState(outState)
    }

    override fun onStop() {
        getSystemService(DisplayManager::class.java).unregisterDisplayListener(displayListener)
        resumeCamera = true
        staticGeneration.incrementAndGet()
        testing = false
        stopCamera()
        super.onStop()
    }

    override fun onDestroy() {
        settingsDialog?.dismiss()
        stopRecording()
        recordingHandler.removeCallbacks(recordingTicker)
        camera.stop()
        worker.shutdown()
        photoWorker.shutdown()
        super.onDestroy()
    }
}
