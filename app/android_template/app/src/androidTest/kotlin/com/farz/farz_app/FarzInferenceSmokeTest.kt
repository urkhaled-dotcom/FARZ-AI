package com.farz.farz_app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Exercises the same ONNX Runtime classifier used by the release app.
 * The test uses a synthetic image so it is deterministic and needs no network,
 * camera app, gallery app, or external test fixture.
 */
@RunWith(AndroidJUnit4::class)
class FarzInferenceSmokeTest {
    @Test
    fun bundledModelRunsOneImageAndReturnsFiveFiniteScores() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val classifier = FarzClassifier(context)

        assertTrue("The ONNX model must be bundled in native Android assets.", classifier.isModelReady())

        val imageFile = File(context.cacheDir, "farz-inference-smoke.png")
        val bitmap = Bitmap.createBitmap(480, 360, Bitmap.Config.ARGB_8888)
        try {
            val canvas = Canvas(bitmap)
            canvas.drawColor(Color.rgb(245, 247, 242))
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.rgb(40, 130, 80)
                style = Paint.Style.FILL
            }
            canvas.drawCircle(240f, 180f, 105f, paint)
            imageFile.outputStream().use { output ->
                assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
            }
        } finally {
            bitmap.recycle()
        }

        try {
            val result = classifier.classify(imageFile.absolutePath)
            assertNotNull(result["predicted_label"])
            val index = result["predicted_index"] as Number
            assertTrue("Predicted index must be one of five classes.", index.toInt() in 0..4)

            val scores = result["scores"] as? List<*>
            assertNotNull("Classifier must return five class scores.", scores)
            assertEquals(5, scores!!.size)
            val numericScores = scores.map { (it as Number).toDouble() }
            assertTrue("All scores must be finite and within [0,1].",
                numericScores.all { it.isFinite() && it in 0.0..1.0 })
            assertTrue("Softmax scores should sum to approximately one.",
                kotlin.math.abs(numericScores.sum() - 1.0) < 1e-4)

            val latency = (result["latency_ms"] as Number).toDouble()
            assertTrue("Inference latency must be positive.", latency > 0.0)
        } finally {
            imageFile.delete()
        }
    }
}
