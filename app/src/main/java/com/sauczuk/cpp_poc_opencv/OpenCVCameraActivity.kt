package com.sauczuk.cpp_poc_opencv

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.sauczuk.cpp_poc_opencv.databinding.ActivityOpencvCameraBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfRect2d
import org.opencv.core.Point
import org.opencv.core.Rect2d
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class OpenCVCameraActivity : AppCompatActivity() {

    // ── Constantes ────────────────────────────────────────────────────────────
    companion object {
        private const val TAG              = "OpenCVCameraActivity"
        private const val CONF_THRESHOLD   = 0.45f
        private const val NMS_THRESHOLD    = 0.4f
        private const val INPUT_SIZE       = 416.0
        private const val WEIGHTS_FILE     = "yolov4-tiny.weights"
        private const val CFG_FILE         = "yolov4-tiny.cfg"
        private const val NAMES_FILE       = "coco.names"

        init { System.loadLibrary("cpp_poc_opencv") }
    }

    // ── Estado ────────────────────────────────────────────────────────────────
    private lateinit var binding: ActivityOpencvCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private var net: Net? = null
    private var classNames: List<String> = emptyList()
    private val isProcessing = AtomicBoolean(false)

    // ── Permiso ───────────────────────────────────────────────────────────────
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else { Toast.makeText(this, "Permiso de cámara denegado", Toast.LENGTH_LONG).show(); finish() }
    }

    // ── Ciclo de vida ─────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityOpencvCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "Error al inicializar OpenCV", Toast.LENGTH_LONG).show()
            finish(); return
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Cargar modelo YOLO en hilo de fondo
        binding.tvStats.text = "⏳ Cargando modelo YOLO..."
        Thread {
            loadModel()
            runOnUiThread {
                binding.tvStats.text = if (net != null) "✅ Modelo listo"
                else "⚠️ Coloca yolov4-tiny.weights en assets/"
                checkPermissionAndStart()
            }
        }.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    // ── Inicialización del modelo ─────────────────────────────────────────────

    /** Copia un archivo desde assets al directorio interno y devuelve la ruta. */
    private fun copyAssetToFile(name: String): String {
        val dest = File(filesDir, name)
        if (!dest.exists()) {
            assets.open(name).use { src -> dest.outputStream().use { dst -> src.copyTo(dst) } }
        }
        return dest.absolutePath
    }

    private fun loadModel() {
        // Cargar nombres de clases COCO
        try {
            classNames = assets.open(NAMES_FILE).bufferedReader().readLines()
                .filter { it.isNotBlank() }
            Log.d(TAG, "Clases cargadas: ${classNames.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando $NAMES_FILE", e)
        }

        // Verificar que los pesos existan en assets
        if (!assets.list("").orEmpty().contains(WEIGHTS_FILE)) {
            Log.w(TAG, "$WEIGHTS_FILE no encontrado en assets/. " +
                "Descárgalo de: https://github.com/AlexeyAB/darknet/releases/download/" +
                "darknet_yolo_v4_pre/yolov4-tiny.weights")
            return
        }

        try {
            val cfg     = copyAssetToFile(CFG_FILE)
            val weights = copyAssetToFile(WEIGHTS_FILE)
            net = Dnn.readNetFromDarknet(cfg, weights).also {
                it.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
                it.setPreferableTarget(Dnn.DNN_TARGET_CPU)
            }
            Log.d(TAG, "Modelo YOLOv4-tiny cargado correctamente")
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando modelo YOLO", e)
            net = null
        }
    }

    // ── Cámara ────────────────────────────────────────────────────────────────
    private fun checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build()
                .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

            val analyzer = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { img -> processFrame(img) } }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer)
            } catch (e: Exception) {
                Log.e(TAG, "Error al vincular cámara", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── Procesamiento de frames ───────────────────────────────────────────────

    private fun processFrame(image: ImageProxy) {
        // Descartar frame si el anterior aún se está procesando
        if (!isProcessing.compareAndSet(false, true)) { image.close(); return }

        try {
            val bitmap   = image.toBitmap()
            val t0       = System.currentTimeMillis()
            val currentNet = net

            val (resultBitmap, count) = if (currentNet != null) {
                runYolo(currentNet, bitmap)
            } else {
                // Sin modelo: mostrar imagen original con aviso
                Pair(bitmap, -1)
            }

            val ms = System.currentTimeMillis() - t0
            runOnUiThread {
                binding.detectionImage.setImageBitmap(resultBitmap)
                binding.tvStats.text = if (count >= 0)
                    "⏱ ${ms}ms  |  🎯 $count obj"
                else
                    "⚠️ Agrega yolov4-tiny.weights en assets/"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error procesando frame", e)
        } finally {
            isProcessing.set(false)
            image.close()
        }
    }

    /**
     * Pipeline YOLO completo:
     *  1. Bitmap → Mat (OpenCV Java)
     *  2. blobFromImage → forward pass (OpenCV DNN)
     *  3. Parse detecciones + NMS (OpenCV Java)
     *  4. applySpotlightNative → efecto spotlight (C++ / JNI)
     *  5. putText con etiquetas (OpenCV Java)
     */
    private fun runYolo(net: Net, bitmap: Bitmap): Pair<Bitmap, Int> {
        val W = bitmap.width
        val H = bitmap.height

        // 1. Bitmap → Mat RGB
        val rgbaMat = Mat(); Utils.bitmapToMat(bitmap, rgbaMat)
        val rgbMat  = Mat(); Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB)

        // 2. Preprocesar y ejecutar la red
        val blob = Dnn.blobFromImage(
            rgbMat, 1.0 / 255.0, Size(INPUT_SIZE, INPUT_SIZE),
            Scalar(0.0, 0.0, 0.0), true, false
        )
        net.setInput(blob)

        val outputs = ArrayList<Mat>()
        net.forward(outputs, net.unconnectedOutLayersNames)

        // 3. Parsear detecciones
        val boxes       = ArrayList<Rect2d>()
        val scores      = ArrayList<Float>()
        val classIds    = ArrayList<Int>()

        for (output in outputs) {
            for (i in 0 until output.rows()) {
                val objectness = output.get(i, 4)[0].toFloat()
                if (objectness < 0.1f) continue

                var maxScore = 0f; var maxClass = 0
                for (j in 5 until output.cols()) {
                    val s = output.get(i, j)[0].toFloat()
                    if (s > maxScore) { maxScore = s; maxClass = j - 5 }
                }

                val conf = objectness * maxScore
                if (conf < CONF_THRESHOLD) continue

                val cx = output.get(i, 0)[0] * W
                val cy = output.get(i, 1)[0] * H
                val bw = output.get(i, 2)[0] * W
                val bh = output.get(i, 3)[0] * H
                boxes.add(Rect2d(
                    (cx - bw / 2).coerceAtLeast(0.0),
                    (cy - bh / 2).coerceAtLeast(0.0),
                    bw.coerceAtMost(W.toDouble()),
                    bh.coerceAtMost(H.toDouble())
                ))
                scores.add(conf)
                classIds.add(maxClass)
            }
        }

        // 4. NMS (Non-Maximum Suppression) para eliminar duplicados
        val nmsIndices = MatOfInt()
        if (boxes.isNotEmpty()) {
            Dnn.NMSBoxes(
                MatOfRect2d(*boxes.toTypedArray()),
                MatOfFloat(*scores.toFloatArray()),
                CONF_THRESHOLD, NMS_THRESHOLD, nmsIndices
            )
        }
        val kept = nmsIndices.toArray()

        // 5. Construir array de cajas [x1,y1,x2,y2, ...] para el JNI C++
        val boxArray = IntArray(kept.size * 4)
        kept.forEachIndexed { i, idx ->
            val b = boxes[idx]
            boxArray[i * 4 + 0] = b.x.toInt()
            boxArray[i * 4 + 1] = b.y.toInt()
            boxArray[i * 4 + 2] = (b.x + b.width).toInt().coerceAtMost(W - 1)
            boxArray[i * 4 + 3] = (b.y + b.height).toInt().coerceAtMost(H - 1)
        }

        // 6. ══ C++ NATIVO: efecto spotlight ══
        //    Oscurece el fondo y dibuja bordes verdes en los objetos detectados
        val pixelsIn = IntArray(W * H)
        bitmap.getPixels(pixelsIn, 0, W, 0, 0, W, H)
        val pixelsOut = applySpotlightNative(pixelsIn, W, H, boxArray)

        val spotBitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        spotBitmap.setPixels(pixelsOut, 0, W, 0, 0, W, H)

        // 7. Dibujar etiquetas de texto con OpenCV Java
        val resultMat = Mat(); Utils.bitmapToMat(spotBitmap, resultMat)
        kept.forEach { idx ->
            val b     = boxes[idx]
            val label = buildString {
                if (classIds[idx] < classNames.size) append(classNames[classIds[idx]])
                else append("obj${classIds[idx]}")
                append("  ${"%.0f".format(scores[idx] * 100)}%")
            }
            Imgproc.putText(
                resultMat, label,
                Point(b.x, (b.y - 6).coerceAtLeast(12.0)),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.55,
                Scalar(0.0, 255.0, 100.0), 2
            )
        }

        val finalBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(resultMat, finalBitmap)

        // Liberar Mats
        rgbaMat.release(); rgbMat.release(); blob.release()
        outputs.forEach { it.release() }; resultMat.release()

        return Pair(finalBitmap, kept.size)
    }

    // ── JNI ───────────────────────────────────────────────────────────────────

    /**
     * Función C++ nativa que aplica un efecto "spotlight":
     *  - Oscurece todos los píxeles fuera de las cajas detectadas (×0.25)
     *  - Mantiene los píxeles dentro de las cajas en su brillo original
     *  - Dibuja un borde verde de 3px alrededor de cada caja
     *
     * @param pixels  Array ARGB del frame completo
     * @param width   Ancho de la imagen
     * @param height  Alto de la imagen
     * @param boxes   Array plano [x1,y1,x2,y2, x1,y1,x2,y2, ...] con las cajas NMS
     * @return        Array ARGB con el efecto aplicado
     */
    external fun applySpotlightNative(
        pixels: IntArray, width: Int, height: Int, boxes: IntArray
    ): IntArray
}

