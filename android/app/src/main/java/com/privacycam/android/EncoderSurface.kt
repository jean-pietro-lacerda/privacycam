package com.privacycam.android

import android.graphics.Bitmap
import android.opengl.EGL14
import android.opengl.EGLExt
import android.opengl.GLES20
import android.opengl.GLUtils
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal class EncoderSurface(
    private val input: Surface,
    private val width: Int,
    private val height: Int
) : AutoCloseable {
    private var display = EGL14.EGL_NO_DISPLAY
    private var context = EGL14.EGL_NO_CONTEXT
    private var surface = EGL14.EGL_NO_SURFACE
    private var program = 0
    private val texture = IntArray(1)
    private var uploaded = false
    private val vertices = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f));
        position(0)
    }
    private val coordinates = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
        put(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f));
        position(0)
    }

    init {
        try {
            display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            check(display != EGL14.EGL_NO_DISPLAY && EGL14.eglInitialize(display, IntArray(2), 0, IntArray(2), 0)) { "Falha EGL." }
            val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
            val attributes = intArrayOf(
                EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT, 0x3142, 1, EGL14.EGL_NONE
            )
            val count = IntArray(1)
            check(EGL14.eglChooseConfig(display, attributes, 0, configs, 0, 1, count, 0) && count[0] > 0) { "Surface de gravação indisponível." }
            context = EGL14.eglCreateContext(
                display, configs[0], EGL14.EGL_NO_CONTEXT,
                intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0
            )
            surface = EGL14.eglCreateWindowSurface(display, configs[0], input, intArrayOf(EGL14.EGL_NONE), 0)
            check(context != EGL14.EGL_NO_CONTEXT && surface != EGL14.EGL_NO_SURFACE && EGL14.eglMakeCurrent(display, surface, surface, context)) { "Falha ao inicializar superfície do encoder." }
            val vertex = shader(GLES20.GL_VERTEX_SHADER, "attribute vec4 position; attribute vec2 uv; varying vec2 tex; void main(){gl_Position=position;tex=uv;}")
            val fragment = shader(GLES20.GL_FRAGMENT_SHADER, "precision mediump float; varying vec2 tex; uniform sampler2D image; void main(){gl_FragColor=texture2D(image,tex);}")
            program = GLES20.glCreateProgram()
            GLES20.glAttachShader(program, vertex)
            GLES20.glAttachShader(program, fragment)
            GLES20.glLinkProgram(program)
            GLES20.glDeleteShader(vertex)
            GLES20.glDeleteShader(fragment)
            val linked = IntArray(1)
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0)
            check(linked[0] != 0) { "Falha no renderizador do encoder." }
            GLES20.glGenTextures(1, texture, 0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture[0])
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        } catch (error: Throwable) {
            close()
            throw error
        }
    }

    private fun shader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val status = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            GLES20.glDeleteShader(shader)
            error("Falha ao preparar renderização do frame processado.")
        }
        return shader
    }

    fun draw(bitmap: Bitmap, presentationNs: Long) {
        check(bitmap.width == width && bitmap.height == height && !bitmap.isRecycled)
        GLES20.glViewport(0, 0, width, height)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture[0])
        if (uploaded) GLUtils.texSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, bitmap)
        else {
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
            uploaded = true
        }
        val position = GLES20.glGetAttribLocation(program, "position")
        val uv = GLES20.glGetAttribLocation(program, "uv")
        GLES20.glEnableVertexAttribArray(position)
        GLES20.glEnableVertexAttribArray(uv)
        GLES20.glVertexAttribPointer(position, 2, GLES20.GL_FLOAT, false, 0, vertices)
        GLES20.glVertexAttribPointer(uv, 2, GLES20.GL_FLOAT, false, 0, coordinates)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "image"), 0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        check(GLES20.glGetError() == GLES20.GL_NO_ERROR) { "Falha ao enviar imagem anonimizada ao encoder." }
        check(EGLExt.eglPresentationTimeANDROID(display, surface, presentationNs) && EGL14.eglSwapBuffers(display, surface)) { "Falha na superfície H.264." }
    }

    override fun close() {
        if (display != EGL14.EGL_NO_DISPLAY) {
            if (context != EGL14.EGL_NO_CONTEXT) {
                GLES20.glDeleteTextures(1, texture, 0)
                if (program != 0) GLES20.glDeleteProgram(program)
            }
            EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            if (surface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, surface)
            if (context != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, context)
            EGL14.eglReleaseThread()
            EGL14.eglTerminate(display)
        }
        input.release()
        display = EGL14.EGL_NO_DISPLAY
        context = EGL14.EGL_NO_CONTEXT
        surface = EGL14.EGL_NO_SURFACE
    }
}
