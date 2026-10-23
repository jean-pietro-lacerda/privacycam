package com.privacycam.android

import android.content.ContentValues
import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.IOException
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.UUID

internal class VideoStore(context: Context) {
    private val resolver = context.applicationContext.contentResolver

    fun save(file: File, width: Int, height: Int, audio: Boolean): Uri {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
            var videoFound = false
            var audioFound = false
            for (track in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(track)
                when (format.getString(MediaFormat.KEY_MIME)) {
                    MediaFormat.MIMETYPE_VIDEO_AVC -> {
                        check(format.getInteger(MediaFormat.KEY_WIDTH) == width && format.getInteger(MediaFormat.KEY_HEIGHT) == height)
                        check(format.containsKey(MediaFormat.KEY_DURATION) && format.getLong(MediaFormat.KEY_DURATION) > 0) { "Gravação muito curta ou inválida." }
                        videoFound = true
                    }
                    MediaFormat.MIMETYPE_AUDIO_AAC -> audioFound = true
                }
            }
            check(videoFound && audioFound == audio) { "Trilhas do MP4 incompletas." }
        } finally {
            extractor.release()
        }
        val stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS"))
        val name = "PrivacyCam_${stamp}_${UUID.randomUUID().toString().take(8)}.mp4"
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.WIDTH, width)
            put(MediaStore.Video.Media.HEIGHT, height)
        }
        var uri: Uri? = null
        var temporary: File? = null
        var legacyFile: File? = null
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/PrivacyCam/")
                values.put(MediaStore.Video.Media.IS_PENDING, 1)
                uri = resolver.insert(MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
                    ?: throw IOException("Falha ao cadastrar vídeo.")
                val output = resolver.openOutputStream(uri, "w") ?: throw IOException("Falha ao abrir vídeo para escrita.")
                output.use { target -> file.inputStream().use { it.copyTo(target) } }
                val input = resolver.openFileDescriptor(uri, "r") ?: throw IOException("Falha ao ler vídeo salvo.")
                input.use { check(it.statSize == file.length()) { "Vídeo salvo incompleto." } }
                val published = ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }
                check(resolver.update(uri, published, null, null) == 1) { "Falha ao publicar vídeo." }
            } else {
                val directory = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "PrivacyCam"
                )
                check(directory.isDirectory || directory.mkdirs()) { "Pasta de vídeos indisponível." }
                val staged = File.createTempFile(".PrivacyCam_", ".pending", directory)
                temporary = staged
                staged.outputStream().use { target -> file.inputStream().use { it.copyTo(target) } }
                check(staged.length() == file.length()) { "Vídeo salvo incompleto." }
                val destination = File(directory, name)
                check(!destination.exists() && staged.renameTo(destination)) { "Falha ao concluir vídeo." }
                legacyFile = destination
                values.put(MediaStore.Video.Media.DATA, destination.absolutePath)
                uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                    ?: throw IOException("Falha ao cadastrar vídeo.")
            }
            return uri
        } catch (error: Throwable) {
            uri?.let {
                try {
                    check(resolver.delete(it, null, null) == 1) { "Falha ao remover vídeo incompleto." }
                } catch (cleanup: Exception) {
                    error.addSuppressed(cleanup)
                }
            }
            legacyFile?.let { if (it.exists() && !it.delete()) error.addSuppressed(IOException("Falha ao remover MP4 incompleto.")) }
            throw error
        } finally {
            temporary?.let { if (it.exists()) it.delete() }
        }
    }
}
