package com.farz.farz_app

import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {
    companion object {
        private const val CHANNEL = "farz/inference"
        private const val TAG = "FARZ-MainActivity"
        private const val MODEL_ASSET = "models/farz_model.onnx"
    }

    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    // Defer construction of the classifier (and any ONNX Runtime class/native
    // initialization) until the user actually requests an inference.
    private val classifier: FarzClassifier by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        FarzClassifier(applicationContext)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "isModelReady" -> inferenceExecutor.execute {
                        // Check the Android asset directly. This deliberately
                        // does not touch FarzClassifier or ONNX Runtime at startup.
                        val ready = try {
                            assets.open(MODEL_ASSET).use { stream -> stream.read() >= 0 }
                        } catch (error: Throwable) {
                            Log.e(TAG, "Unable to inspect bundled model asset.", error)
                            false
                        }
                        mainHandler.post { result.success(ready) }
                    }

                    "classifyImage" -> {
                        val path = call.argument<String>("path")
                        if (path.isNullOrBlank()) {
                            result.error("INVALID_IMAGE_PATH", "A local image path is required.", null)
                        } else {
                            inferenceExecutor.execute {
                                try {
                                    val output = classifier.classify(path)
                                    mainHandler.post { result.success(output) }
                                } catch (error: Throwable) {
                                    Log.e(TAG, "On-device inference failed.", error)
                                    mainHandler.post {
                                        val code = when {
                                            error.message?.contains("MODEL_NOT_FOUND") == true -> "MODEL_NOT_FOUND"
                                            error is LinkageError -> "NATIVE_RUNTIME_UNAVAILABLE"
                                            else -> "INFERENCE_FAILED"
                                        }
                                        result.error(code, error.message ?: "On-device inference failed.", null)
                                    }
                                }
                            }
                        }
                    }

                    else -> result.notImplemented()
                }
            }
    }

    override fun onDestroy() {
        inferenceExecutor.shutdown()
        super.onDestroy()
    }
}
