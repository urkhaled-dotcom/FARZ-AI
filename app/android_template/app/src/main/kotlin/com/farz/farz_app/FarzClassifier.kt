package com.farz.farz_app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.system.Os
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

/** Runs the trained FARZ ONNX model locally with bounded image memory use. */
class FarzClassifier(private val context: Context) {
    companion object {
        private const val TAG = "FARZ-Classifier"
        private const val MODEL_ASSET = "models/farz_model.onnx"
        private const val IMAGE_SIZE = 224
        private const val MAX_DECODE_EDGE = 512
        private val CLASS_NAMES = listOf("Glass", "Paper", "Cardboard", "Plastic", "Metal")
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val environment: OrtEnvironment by lazy { createOfflineEnvironment() }
    @Volatile private var session: OrtSession? = null
    @Volatile private var sessionOptions: OrtSession.SessionOptions? = null
    @Volatile private var inputName: String? = null

    private fun createOfflineEnvironment(): OrtEnvironment {
        try {
            Os.setenv("ORT_DISABLE_TELEMETRY", "1", true)
        } catch (error: Exception) {
            Log.w(TAG, "Could not set telemetry environment flag.", error)
        }
        val env = OrtEnvironment.getEnvironment()
        try {
            env.setTelemetry(false)
        } catch (error: Exception) {
            Log.w(TAG, "Could not explicitly disable ONNX Runtime telemetry.", error)
        }
        return env
    }

    fun isModelReady(): Boolean = try {
        context.assets.open(MODEL_ASSET).use { stream -> stream.read() >= 0 }
    } catch (error: Exception) {
        Log.e(TAG, "Model asset is missing or unreadable: " + MODEL_ASSET, error)
        false
    }

    @Synchronized
    private fun getSession(): OrtSession {
        session?.let { return it }
        Log.i(TAG, "Preparing local ONNX session.")
        val modelPath = copyModelToFiles()
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(1)
        options.setInterOpNumThreads(1)
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        val loaded = try {
            environment.createSession(modelPath, options)
        } catch (error: Throwable) {
            options.close()
            throw error
        }
        val name = loaded.inputNames.firstOrNull()
        if (name == null) {
            loaded.close()
            options.close()
            throw IllegalStateException("ONNX model has no input node.")
        }
        sessionOptions = options
        inputName = name
        session = loaded
        Log.i(TAG, "ONNX model loaded. Input node: " + name)
        return loaded
    }

    private fun copyModelToFiles(): String {
        val modelFile = File(context.filesDir, "farz_model.onnx")
        if (!modelFile.exists() || modelFile.length() == 0L) {
            val temporary = File(context.filesDir, "farz_model.onnx.tmp")
            try {
                context.assets.open(MODEL_ASSET).use { input ->
                    temporary.outputStream().use { output -> input.copyTo(output) }
                }
            } catch (error: Exception) {
                temporary.delete()
                throw IllegalStateException(
                    "MODEL_NOT_FOUND: add app/assets/models/farz_model.onnx and rebuild the APK.",
                    error
                )
            }
            if (temporary.length() == 0L) {
                temporary.delete()
                throw IllegalStateException("MODEL_NOT_FOUND: empty ONNX model.")
            }
            if (!temporary.renameTo(modelFile)) {
                temporary.copyTo(modelFile, overwrite = true)
                temporary.delete()
            }
        }
        return modelFile.absolutePath
    }

    fun classify(imagePath: String): Map<String, Any> {
        val start = System.nanoTime()
        val imageFile = File(imagePath)
        require(imageFile.isFile && imageFile.canRead() && imageFile.length() > 0L) {
            "IMAGE_UNREADABLE: the selected image cannot be read."
        }
        Log.i(TAG, "Classification requested for " + imageFile.length() + " bytes.")
        val input = preprocess(imageFile)
        Log.i(TAG, "Image preprocessing complete; creating ONNX tensor.")
        val ortSession = getSession()
        val tensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(input),
            longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        )
        val results = try {
            Log.i(TAG, "Starting ONNX CPU inference.")
            ortSession.run(mapOf(inputName!! to tensor))
        } finally {
            tensor.close()
        }
        val logits = try {
            val outputTensor = results[0].value as? OnnxTensor
                ?: throw IllegalStateException("ONNX model output is not a tensor.")
            val outputBuffer = outputTensor.floatBuffer
            FloatArray(outputBuffer.remaining()).also { outputBuffer.get(it) }
        } finally {
            results.close()
        }
        if (logits.size != CLASS_NAMES.size || logits.any { !it.isFinite() }) {
            throw IllegalStateException("ONNX model returned invalid output values (" + logits.size + ").")
        }
        val probabilities = softmax(logits)
        val predictedIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val latencyMs = (System.nanoTime() - start) / 1_000_000.0
        Log.i(TAG, "Inference completed in " + latencyMs.toLong() + " ms.")
        return mapOf(
            "predicted_index" to predictedIndex,
            "predicted_label" to CLASS_NAMES[predictedIndex],
            "scores" to probabilities.map { it.toDouble() },
            "latency_ms" to latencyMs
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxValue = logits.maxOrNull() ?: 0f
        val exponentials = FloatArray(logits.size) {
            exp((logits[it] - maxValue).toDouble()).toFloat()
        }
        val total = exponentials.sum().coerceAtLeast(1e-12f)
        return FloatArray(logits.size) { exponentials[it] / total }
    }

    /**
     * Decode at a bounded size, apply EXIF orientation, and draw directly into
     * the 224x224 padded output. This avoids allocating a second full-size
     * square bitmap and guarantees that temporary bitmaps are recycled.
     */
    private fun preprocess(file: File): FloatArray {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IllegalArgumentException("IMAGE_DECODE_FAILED: invalid image dimensions.")
        }

        var sampleSize = 1
        while (max(bounds.outWidth / sampleSize, bounds.outHeight / sampleSize) > MAX_DECODE_EDGE) {
            sampleSize *= 2
        }
        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
        }
        Log.i(TAG, "Decoding " + bounds.outWidth + "x" + bounds.outHeight + ", sample=" + sampleSize + ".")
        val source = BitmapFactory.decodeFile(file.absolutePath, decodeOptions)
            ?: throw IllegalArgumentException("IMAGE_DECODE_FAILED: Android could not decode this image.")

        var oriented: Bitmap? = null
        var target: Bitmap? = null
        try {
            oriented = applyExifOrientation(source, file.absolutePath)
            val width = oriented.width
            val height = oriented.height
            val side = max(width, height).toFloat()
            val scale = IMAGE_SIZE.toFloat() / side
            val drawWidth = width * scale
            val drawHeight = height * scale
            val left = (IMAGE_SIZE - drawWidth) / 2f
            val top = (IMAGE_SIZE - drawHeight) / 2f

            target = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(target)
            canvas.drawColor(Color.rgb(245, 247, 242))
            val destination = RectF(left, top, left + drawWidth, top + drawHeight)
            val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG or Paint.ANTI_ALIAS_FLAG)
            canvas.drawBitmap(oriented, null, destination, paint)

            val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
            target.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
            val plane = IMAGE_SIZE * IMAGE_SIZE
            val input = FloatArray(3 * plane)
            for (i in pixels.indices) {
                val pixel = pixels[i]
                input[i] = (Color.red(pixel) / 255f - MEAN[0]) / STD[0]
                input[plane + i] = (Color.green(pixel) / 255f - MEAN[1]) / STD[1]
                input[2 * plane + i] = (Color.blue(pixel) / 255f - MEAN[2]) / STD[2]
            }
            return input
        } finally {
            target?.let { if (it !== source && it !== oriented && !it.isRecycled) it.recycle() }
            oriented?.let { if (it !== source && !it.isRecycled) it.recycle() }
            if (!source.isRecycled) source.recycle()
        }
    }

    private fun applyExifOrientation(bitmap: Bitmap, path: String): Bitmap {
        val orientation = try {
            ExifInterface(path).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        } catch (_: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.preScale(-1f, 1f)
                matrix.postRotate(90f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.preScale(-1f, 1f)
                matrix.postRotate(270f)
            }
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
