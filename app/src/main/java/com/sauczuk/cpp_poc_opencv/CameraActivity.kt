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
import com.sauczuk.cpp_poc_opencv.databinding.ActivityCameraBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraExecutor: ExecutorService

    // Lanzador de solicitud de permiso de cámara
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Permiso de cámara denegado", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Inicializar OpenCV
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "Error al inicializar OpenCV", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        Log.d(TAG, "OpenCV inicializado correctamente: ${org.opencv.core.Core.VERSION}")

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Verificar/solicitar permiso de cámara
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview en vivo
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            // Análisis de frames para procesamiento con OpenCV
            val imageAnalyzer = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error al vincular la cámara", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Procesa cada frame de la cámara:
     * 1. Convierte el ImageProxy a Bitmap (RGBA)
     * 2. Usa OpenCV Java API: convierte a escala de grises y aplica Canny
     * 3. Llama a la función C++ nativa para mostrar el pipeline JNI
     * 4. Muestra el resultado en el ImageView
     */
    private fun processFrame(image: ImageProxy) {
        try {
            // Obtener Bitmap del frame (RGBA_8888 garantizado por la configuración)
            val bitmap: Bitmap = image.toBitmap()

            // === OpenCV (Java API) ===
            val rgbaMat = Mat()
            Utils.bitmapToMat(bitmap, rgbaMat)

            val grayMat = Mat()
            Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY)

            val edgesMat = Mat()
            Imgproc.Canny(grayMat, edgesMat, 80.0, 180.0)

            val resultBitmap = Bitmap.createBitmap(
                edgesMat.cols(), edgesMat.rows(), Bitmap.Config.ARGB_8888
            )
            Utils.matToBitmap(edgesMat, resultBitmap)

            // Liberar Mats para evitar memory leaks
            rgbaMat.release()
            grayMat.release()
            edgesMat.release()

            // === Pipeline C++ nativo (JNI) ===
            // Pasa los píxeles RGBA al código C++ para procesamiento adicional
            val width = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            val nativeResult = processFrameNative(pixels, width, height)
            Log.v(TAG, "C++ procesó ${nativeResult.size} píxeles (${width}x${height})")

            // Mostrar resultado de OpenCV en el ImageView
            runOnUiThread {
                binding.processedImage.setImageBitmap(resultBitmap)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error procesando frame", e)
        } finally {
            image.close()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    /**
     * Función nativa C++: recibe píxeles RGBA, aplica conversión a escala de grises,
     * y retorna el array procesado. Demuestra el pipeline Kotlin → JNI → C++.
     */
    external fun processFrameNative(pixels: IntArray, width: Int, height: Int): IntArray

    companion object {
        private const val TAG = "CameraActivity"

        init {
            System.loadLibrary("cpp_poc_opencv")
        }
    }
}

