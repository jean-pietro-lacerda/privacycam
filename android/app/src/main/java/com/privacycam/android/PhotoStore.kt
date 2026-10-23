package com.privacycam.android

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.IOException
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.UUID

class PhotoStore(context: Context) {
    private val resolver = context.applicationContext.contentResolver

    fun save(image: Bitmap): Uri {
        check(!image.isRecycled && image.width > 0 && image.height > 0) { "Imagem inválida." }
        val stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS"))
        val name = "PrivacyCam_${stamp}_${UUID.randomUUID().toString().take(8)}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.WIDTH, image.width)
            put(MediaStore.Images.Media.HEIGHT, image.height)
        }
        var uri: Uri? = null
        var temporary: File? = null
        var legacyFile: File? = null
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/PrivacyCam/")
                values.put(MediaStore.Images.Media.IS_PENDING, 1)
                uri = resolver.insert(
                    MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values
                ) ?: throw IOException("Falha ao criar entrada no MediaStore.")
                resolver.openOutputStream(uri, "w")?.use { output ->
                    if (!image.compress(Bitmap.CompressFormat.JPEG, 95, output))
                        throw IOException("Falha ao codificar JPEG.")
                } ?: throw IOException("Falha ao abrir arquivo para escrita.")
            } else {
                val directory = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "PrivacyCam"
                )
                if (!directory.isDirectory && !directory.mkdirs())
                    throw IOException("Pasta de fotografias indisponível.")
                val staged = File.createTempFile(".PrivacyCam_", ".pending", directory)
                temporary = staged
                staged.outputStream().use { output ->
                    if (!image.compress(Bitmap.CompressFormat.JPEG, 95, output))
                        throw IOException("Falha ao codificar JPEG.")
                }
                val destination = File(directory, name)
                if (destination.exists() || !staged.renameTo(destination))
                    throw IOException("Falha ao concluir o arquivo JPEG.")
                legacyFile = destination
                values.put(MediaStore.Images.Media.DATA, destination.absolutePath)
                uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: throw IOException("Falha ao cadastrar foto no MediaStore.")
            }
            val saved = uri
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            val input = resolver.openInputStream(saved)
                ?: throw IOException("Falha ao abrir a foto salva para leitura.")
            input.use { BitmapFactory.decodeStream(it, null, bounds) }
            if (bounds.outWidth != image.width || bounds.outHeight != image.height || bounds.outMimeType != "image/jpeg")
                throw IOException("JPEG salvo inválido ou incompleto.")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val published = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
                if (resolver.update(saved, published, null, null) != 1)
                    throw IOException("Falha ao publicar foto na galeria.")
            }
            return saved
        } catch (error: Throwable) {
            uri?.let { incomplete ->
                try {
                    if (resolver.delete(incomplete, null, null) != 1)
                        throw IOException("Não foi possível remover a entrada incompleta.")
                } catch (cleanup: Exception) {
                    error.addSuppressed(cleanup)
                }
            }
            legacyFile?.let { file ->
                if (file.exists() && !file.delete())
                    error.addSuppressed(IOException("Não foi possível remover o JPEG incompleto."))
            }
            throw error
        } finally {
            temporary?.let { if (it.exists()) it.delete() }
        }
    }
}
