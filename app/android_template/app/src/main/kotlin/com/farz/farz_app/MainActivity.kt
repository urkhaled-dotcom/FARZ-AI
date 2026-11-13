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
    }

    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var classifier: FarzClassifier

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        classifier = FarzClassifier(applicationContext)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "isModelReady" -> inferenceExecutor.execute {
                        val ready = try {
                            classifier.isModelReady()
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
