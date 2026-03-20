package com.sauczuk.cpp_poc_opencv

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.sauczuk.cpp_poc_opencv.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Llamadas a funciones nativas C++
//        binding.sampleText.text = stringFromJNI()
//        binding.sampleText2.text = string2FromJNI()

        // Botón para abrir CameraActivity con OpenCV Canny
        binding.btnCamera.setOnClickListener {
            startActivity(Intent(this, CameraActivity::class.java))
        }

        // Botón para abrir OpenCVCameraActivity con YOLO + C++ Spotlight
        binding.btnYolo.setOnClickListener {
            startActivity(Intent(this, OpenCVCameraActivity::class.java))
        }
    }

    external fun stringFromJNI(): String
    external fun string2FromJNI(): String

    companion object {
        init {
            System.loadLibrary("cpp_poc_opencv")
        }
    }
}


/**
 * Java_ | com_sauczuk_cpp_1poc_1opencv | MainActivity | stringFromJNI
   ↑              ↑                         ↑               ↑
 prefijo       paquete                    clase          método Kotlin
              (. → _)
              (- → _1)
 */
