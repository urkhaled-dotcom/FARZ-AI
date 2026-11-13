package com.farz.farz_app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import android.system.Os
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

/** Executes the trained FARZ ONNX model locally with ONNX Runtime CPU. */
class FarzClassifier(private val context: Context) {
    companion object {
        private const val TAG = "FARZ-Classifier"
        private const val MODEL_ASSET = "models/farz_model.onnx"
        private const val IMAGE_SIZE = 224
        private const val MAX_DECODE_EDGE = 1024
        private val CLASS_NAMES = listOf("Glass", "Paper", "Cardboard", "Plastic", "Metal")
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val environment: OrtEnvironment by lazy { createOfflineEnvironment() }
    @Volatile private var session: OrtSession? = null
    @Volatile private var sessionOptions: OrtSession.SessionOptions? = null
    @Volatile private var inputName: String? = null

    private fun createOfflineEnvironment(): OrtEnvironment {
        // Disable native telemetry before ONNX Runtime initializes.
        try {
            Os.setenv("ORT_DISABLE_TELEMETRY", "1", true)
        } catch (error: Exception) {
            Log.w(TAG, "Could not set telemetry environment flag; API opt-out will still be applied.", error)
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
        Log.e(TAG, "Model asset is missing or unreadable: $MODEL_ASSET", error)
        false
    }

    @Synchronized
    private fun getSession(): OrtSession {
        session?.let { return it }
        val modelPath = copyModelToFiles()
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(2)
        options.setInterOpNumThreads(1)
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
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
        // Retain SessionOptions for at least as long as the associated OrtSession.
        sessionOptions = options
        inputName = name
        session = loaded
        Log.i(TAG, "ONNX model loaded successfully. Input node: $name")
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
            if (temporary.length() == 0L) throw IllegalStateException("MODEL_NOT_FOUND: empty ONNX model.")
            if (!temporary.renameTo(modelFile)) {
                temporary.copyTo(modelFile, overwrite = true)
                temporary.delete()
            }
        }
        return modelFile.absolutePath
    }

    fun classify(imagePath: String): Map<String, Any> {
        val start = System.nanoTime()
        val ortSession = getSession()
        val input = preprocess(imagePath)
        val tensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(input),
            longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        )
        val results = try {
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
        if (logits.size != CLASS_NAMES.size) {
            throw IllegalStateException("Expected 5 class scores, received ${logits.size}.")
        }
        val probabilities = softmax(logits)
        val predictedIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val latencyMs = (System.nanoTime() - start) / 1_000_000.0
        return mapOf(
            "predicted_index" to predictedIndex,
            "predicted_label" to CLASS_NAMES[predictedIndex],
            "scores" to probabilities.map { it.toDouble() },
            "latency_ms" to latencyMs
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxValue = logits.maxOrNull() ?: 0f
        val exponentials = FloatArray(logits.size) { exp((logits[it] - maxValue).toDouble()).toFloat() }
        val total = exponentials.sum().coerceAtLeast(1e-12f)
        return FloatArray(logits.size) { exponentials[it] / total }
    }

    private fun preprocess(path: String): FloatArray {
        // Decode camera originals at a bounded working size to avoid OOM on phones
        // with limited RAM (notably older Android 10 devices).
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IllegalArgumentException("Could not read selected image dimensions.")
        }
        var sampleSize = 1
        while (max(bounds.outWidth / sampleSize, bounds.outHeight / sampleSize) > MAX_DECODE_EDGE) {
            sampleSize *= 2
        }
        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = android.graphics.Bitmap.Config.ARGB_8888
        }
        Log.i(TAG, "Decoding image ${bounds.outWidth}x${bounds.outHeight} with sample=$sampleSize")
        val source = BitmapFactory.decodeFile(path, decodeOptions)
            ?: throw IllegalArgumentException("Could not decode selected image.")
        val oriented = applyExifOrientation(source, path)
        val width = oriented.width
        val height = oriented.height
        val side = max(width, height)
        val square = android.graphics.Bitmap.createBitmap(side, side, android.graphics.Bitmap.Config.ARGB_8888)
        Canvas(square).apply {
            drawColor(Color.rgb(245, 247, 242))
            // Match PIL SquarePad's integer-floor left/top offsets exactly.
            val left = (side - width) / 2
            val top = (side - height) / 2
            drawBitmap(oriented, left.toFloat(), top.toFloat(),
                Paint(Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG))
        }
        val resized = android.graphics.Bitmap.createScaledBitmap(square, IMAGE_SIZE, IMAGE_SIZE, true)
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        val plane = IMAGE_SIZE * IMAGE_SIZE
        val tensor = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            tensor[i] = (Color.red(pixel) / 255f - MEAN[0]) / STD[0]
            tensor[plane + i] = (Color.green(pixel) / 255f - MEAN[1]) / STD[1]
            tensor[2 * plane + i] = (Color.blue(pixel) / 255f - MEAN[2]) / STD[2]
        }
        if (source !== oriented && !source.isRecycled) source.recycle()
        if (!oriented.isRecycled) oriented.recycle()
        if (!square.isRecycled) square.recycle()
        if (!resized.isRecycled) resized.recycle()
        return tensor
    }

    private fun applyExifOrientation(bitmap: android.graphics.Bitmap, path: String): android.graphics.Bitmap {
        val orientation = try {
            ExifInterface(path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
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
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.preScale(-1f, 1f); matrix.postRotate(90f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.preScale(-1f, 1f); matrix.postRotate(270f) }
            else -> return bitmap
        }
        return android.graphics.Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
