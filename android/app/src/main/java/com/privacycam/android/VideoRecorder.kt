package com.privacycam.android

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTimestamp
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

internal class VideoRecorder(
    context: Context,
    private val width: Int,
    private val height: Int,
    private val requestAudio: Boolean,
    private val onStarted: (Boolean, Long) -> Unit,
    private val onFinished: (Uri?, String?) -> Unit
) {
    private val app = context.applicationContext
    private val handler = Handler(Looper.getMainLooper())
    private val stopping = AtomicBoolean()
    private val accepting = AtomicBoolean()
    private val ending = AtomicLong(Long.MAX_VALUE)
    private val failure = AtomicReference<Throwable?>()
    private val latest = AtomicReference<Frame?>()
    private val wake = Semaphore(0)
    private var origin = 0L

    private data class Frame(val bitmap: Bitmap, val capturedNs: Long)
    private data class Microphone(val record: AudioRecord, val encoder: MediaCodec, val rate: Int)

    fun start() {
        Thread({ run() }, "PrivacyCam-video").start()
    }

    fun offer(bitmap: Bitmap, capturedNs: Long) {
        if (!accepting.get() || stopping.get() || bitmap.isRecycled) return
        if (bitmap.width != width || bitmap.height != height) {
            stop()
            return
        }
        latest.set(Frame(bitmap, capturedNs))
        if (wake.availablePermits() == 0) wake.release()
    }

    fun stop() {
        accepting.set(false)
        ending.compareAndSet(Long.MAX_VALUE, System.nanoTime())
        stopping.set(true)
        wake.release()
    }

    private fun fail(error: Throwable) {
        failure.compareAndSet(null, error)
        stop()
    }

    private fun microphone(): Microphone? {
        if (!requestAudio || app.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return null
        for (rate in intArrayOf(48000, 44100)) {
            var record: AudioRecord? = null
            var encoder: MediaCodec? = null
            try {
                val minimum = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
                check(minimum > 0)
                record = AudioRecord.Builder()
                    .setAudioSource(MediaRecorder.AudioSource.MIC)
                    .setAudioFormat(AudioFormat.Builder().setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_IN_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                    .setBufferSizeInBytes(maxOf(minimum * 4, rate / 5 * 2)).build()
                check(record.state == AudioRecord.STATE_INITIALIZED)
                encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
                val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, rate, 1).apply {
                    setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
                    setInteger(MediaFormat.KEY_BIT_RATE, 64000)
                    setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 4096)
                }
                encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                encoder.start()
                return Microphone(record, encoder, rate)
            } catch (error: Exception) {
                runCatching { record?.release() }
                runCatching { encoder?.release() }
                Log.w("PrivacyCam", "Microfone/AAC indisponível em $rate Hz", error)
            }
        }
        return null
    }

    private fun run() {
        var file: File? = null
        var video: MediaCodec? = null
        var renderer: EncoderSurface? = null
        var audio: Microphone? = null
        var audioThread: Thread? = null
        var muxer: Muxer? = null
        var saved: Uri? = null
        var error: Throwable? = null
        try {
            check(width % 2 == 0 && height % 2 == 0) { "Resolução não suportada pelo encoder H.264." }
            file = File.createTempFile("privacycam-video-", ".mp4", app.cacheDir)
            val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            video = encoder
            check(encoder.codecInfo.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC).videoCapabilities?.isSizeSupported(width, height) == true) { "Resolução H.264 indisponível." }
            val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                setInteger(MediaFormat.KEY_BIT_RATE, 2_000_000)
                setInteger(MediaFormat.KEY_FRAME_RATE, 30)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                if (Build.VERSION.SDK_INT >= 29) setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
            }
            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            renderer = EncoderSurface(encoder.createInputSurface(), width, height)
            encoder.start()
            if (!stopping.get()) audio = microphone()
            check(!stopping.get()) { "Gravação interrompida antes de iniciar." }
            origin = System.nanoTime()
            audio?.let { mic ->
                try {
                    mic.record.startRecording()
                    check(mic.record.recordingState == AudioRecord.RECORDSTATE_RECORDING)
                } catch (unavailable: Exception) {
                    runCatching { mic.record.release() }
                    runCatching { mic.encoder.release() }
                    audio = null
                    Log.w("PrivacyCam", "Áudio desativado: microfone indisponível", unavailable)
                }
            }
            val sink = Muxer(MediaMuxer(file.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4), audio != null)
            muxer = sink
            audio?.let { mic ->
                audioThread = Thread({
                    try { encodeAudio(mic, sink) } catch (problem: Throwable) { fail(problem) }
                    finally { runCatching { mic.record.stop() } }
                }, "PrivacyCam-audio").also { it.start() }
            }
            if (!stopping.get()) {
                accepting.set(true)
                val hasAudio = audio != null
                handler.post { if (!stopping.get()) onStarted(hasAudio, origin) }
            }
            var lastPts = -1L
            while (!stopping.get()) {
                wake.tryAcquire(10, TimeUnit.MILLISECONDS)
                wake.drainPermits()
                drain(encoder, sink, true, false)
                val frame = latest.getAndSet(null) ?: continue
                if (frame.capturedNs < origin) continue
                val pts = (frame.capturedNs - origin) / 1000
                if (pts < 0 || pts <= lastPts) continue
                renderer.draw(frame.bitmap, pts * 1000)
                lastPts = pts
                drain(encoder, sink, true, false)
            }
            accepting.set(false)
            latest.set(null)
            failure.get()?.let { throw it }
            encoder.signalEndOfInputStream()
            drain(encoder, sink, true, true)
            audioThread?.join(5000)
            check(audioThread?.isAlive != true) { "Áudio não finalizou no prazo." }
            failure.get()?.let { throw it }
            sink.finish(maxOf(0L, (ending.get() - origin) / 1000))
            val hadAudio = audio != null
            audio?.record?.release()
            audio?.encoder?.stop()
            audio?.encoder?.release()
            audio = null
            encoder.stop()
            renderer.close()
            renderer = null
            encoder.release()
            video = null
            sink.close()
            muxer = null
            saved = VideoStore(app).save(file, width, height, hadAudio)
        } catch (problem: Throwable) {
            error = problem
            fail(problem)
        } finally {
            accepting.set(false)
            stopping.set(true)
            latest.set(null)
            audioThread?.let {
                if (it.isAlive) runCatching { audio?.record?.stop() }
                it.join(3000)
            }
            runCatching { audio?.record?.release() }
            runCatching { audio?.encoder?.stop() }
            runCatching { audio?.encoder?.release() }
            runCatching { video?.stop() }
            runCatching { renderer?.close() }
            runCatching { video?.release() }
            runCatching { muxer?.close() }
            file?.let { if (it.exists()) it.delete() }
            val result = saved
            val message = error?.let { it.message ?: it.javaClass.simpleName }
            handler.post { onFinished(result, message) }
        }
    }

    private fun encodeAudio(mic: Microphone, sink: Muxer) {
        val samples = ByteArray(2048)
        val timestamp = AudioTimestamp()
        var readFrames = 0L
        var fallbackStart = 0L
        var timestampWarning = false
        var lastPts = -1L
        var lastRead = origin
        val bufferDuration = mic.record.bufferSizeInFrames * 1_000_000_000L / mic.rate
        while (!stopping.get()) {
            val count = mic.record.read(samples, 0, samples.size, AudioRecord.READ_BLOCKING)
            check(count > 0 && count % 2 == 0) { "Falha ao capturar áudio ($count)." }
            val now = System.nanoTime()
            check(readFrames == 0L || now - lastRead <= bufferDuration) { "Captura de áudio atrasada; sincronização não pode ser garantida." }
            lastRead = now
            val frames = count / 2
            if (fallbackStart == 0L) fallbackStart = now - frames * 1_000_000_000L / mic.rate
            val hardwareTimestamp = mic.record.getTimestamp(timestamp, AudioTimestamp.TIMEBASE_MONOTONIC) == AudioRecord.SUCCESS
            if (!hardwareTimestamp && !timestampWarning) {
                Log.w("PrivacyCam", "Timestamp de áudio indisponível; sincronia usa estimativa de leitura até recuperação.")
                timestampWarning = true
            }
            val captureNs = if (hardwareTimestamp)
                timestamp.nanoTime + (readFrames - timestamp.framePosition) * 1_000_000_000L / mic.rate
            else fallbackStart + readFrames * 1_000_000_000L / mic.rate
            readFrames += frames
            val skip = if (captureNs < origin) minOf(frames, ((origin - captureNs) * mic.rate / 1_000_000_000L).toInt()) else 0
            val allowed = minOf(frames.toLong(), ((ending.get() - captureNs).coerceAtLeast(0) / 1_000_000L * mic.rate / 1000)).toInt()
            val usable = (allowed - skip).coerceAtLeast(0)
            if (usable == 0) continue
            val pts = maxOf(lastPts + 1, (captureNs + skip * 1_000_000_000L / mic.rate - origin) / 1000)
            queueAudio(mic.encoder, sink, samples, skip * 2, usable * 2, pts, false)
            lastPts = pts
        }
        mic.record.stop()
        queueAudio(mic.encoder, sink, samples, 0, 0, maxOf(lastPts + 1, 0L, (ending.get() - origin) / 1000), true)
        drain(mic.encoder, sink, false, true)
    }

    private fun queueAudio(codec: MediaCodec, sink: Muxer, bytes: ByteArray, offset: Int, size: Int, pts: Long, eos: Boolean) {
        val deadline = System.nanoTime() + 3_000_000_000L
        while (true) {
            val index = codec.dequeueInputBuffer(10000)
            if (index >= 0) {
                val buffer = codec.getInputBuffer(index) ?: error("Buffer AAC indisponível.")
                buffer.clear()
                check(buffer.remaining() >= size)
                buffer.put(bytes, offset, size)
                codec.queueInputBuffer(index, 0, size, pts, if (eos) MediaCodec.BUFFER_FLAG_END_OF_STREAM else 0)
                if (!eos) drain(codec, sink, false, false)
                return
            }
            drain(codec, sink, false, false)
            check(System.nanoTime() < deadline) { "Encoder AAC bloqueado." }
        }
    }

    private fun drain(codec: MediaCodec, sink: Muxer, video: Boolean, eos: Boolean) {
        val info = MediaCodec.BufferInfo()
        val deadline = System.nanoTime() + 5_000_000_000L
        while (true) {
            val index = codec.dequeueOutputBuffer(info, if (eos) 10000 else 0)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!eos) return
                    check(System.nanoTime() < deadline) { "Encoder não finalizou." }
                }
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> sink.format(video, codec.outputFormat)
                index >= 0 -> {
                    try {
                        if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                            val buffer = codec.getOutputBuffer(index) ?: error("Amostra codificada indisponível.")
                            sink.sample(video, buffer, info)
                        }
                    } finally { codec.releaseOutputBuffer(index, false) }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    private class Muxer(private val output: MediaMuxer, private val audio: Boolean) : AutoCloseable {
        private data class Sample(val video: Boolean, val data: ByteBuffer, val info: MediaCodec.BufferInfo)
        private var videoTrack = -1
        private var audioTrack = -1
        private var started = false
        private var finished = false
        private var pendingBytes = 0
        private val pending = ArrayList<Sample>()
        private var videoSamples = 0L
        private var audioSamples = 0L
        private var lastVideo = -1L
        private var lastAudio = -1L

        @Synchronized fun format(video: Boolean, format: MediaFormat) {
            if (video) {
                check(videoTrack == -1)
                videoTrack = output.addTrack(format)
            } else {
                check(audio && audioTrack == -1)
                audioTrack = output.addTrack(format)
            }
            if (videoTrack >= 0 && (!audio || audioTrack >= 0)) {
                output.start()
                started = true
                pending.sortBy { it.info.presentationTimeUs }
                pending.forEach { write(it.video, it.data, it.info) }
                pending.clear()
                pendingBytes = 0
            }
        }

        @Synchronized fun sample(video: Boolean, data: ByteBuffer, info: MediaCodec.BufferInfo) {
            if (info.presentationTimeUs < 0) return
            if (started) write(video, data, info)
            else {
                check(pending.size < 256 && pendingBytes + info.size <= 2_000_000) { "Trilhas não ficaram prontas a tempo." }
                val bytes = ByteBuffer.allocate(info.size)
                val view = data.duplicate().apply { position(info.offset); limit(info.offset + info.size) }
                bytes.put(view).flip()
                val copy = MediaCodec.BufferInfo().apply { set(0, info.size, info.presentationTimeUs, info.flags) }
                pending.add(Sample(video, bytes, copy))
                pendingBytes += info.size
            }
        }

        private fun write(video: Boolean, data: ByteBuffer, info: MediaCodec.BufferInfo) {
            check(!finished && info.presentationTimeUs >= if (video) lastVideo else lastAudio) { "Timestamp de mídia inválido." }
            output.writeSampleData(if (video) videoTrack else audioTrack, data, info)
            if (video) { lastVideo = info.presentationTimeUs; ++videoSamples }
            else { lastAudio = info.presentationTimeUs; ++audioSamples }
        }

        @Synchronized fun finish(endUs: Long) {
            check(started && videoSamples > 0 && (!audio || audioSamples > 0)) { "Gravação sem trilhas completas." }
            val end = MediaCodec.BufferInfo().apply { set(0, 0, maxOf(lastVideo + 1, endUs), MediaCodec.BUFFER_FLAG_END_OF_STREAM) }
            output.writeSampleData(videoTrack, ByteBuffer.allocate(0), end)
            if (audio) {
                end.presentationTimeUs = maxOf(lastAudio + 1, endUs)
                output.writeSampleData(audioTrack, ByteBuffer.allocate(0), end)
            }
            output.stop()
            finished = true
        }

        @Synchronized override fun close() {
            output.release()
            pending.clear()
        }
    }
}
