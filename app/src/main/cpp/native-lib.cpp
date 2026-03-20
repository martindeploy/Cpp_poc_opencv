#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>


/**
 * OpenCVCameraActivity — Efecto "Spotlight" con YOLO.
 *
 * Recibe los píxeles RGBA del frame y las cajas de detección ya filtradas por NMS.
 * Aplica en C++:
 *   1. Oscurece todos los píxeles que están FUERA de las cajas (×0.25)
 *   2. Conserva el brillo original de los píxeles DENTRO de las cajas
 *   3. Dibuja un borde verde (#00FF00) de 3px alrededor de cada caja
 *
 * Pipeline: Kotlin (CameraX) → OpenCV DNN (YOLO) → JNI → C++ → Kotlin (ImageView)
 *
 * @param pixels  IntArray ARGB del frame completo
 * @param width   Ancho de la imagen en píxeles
 * @param height  Alto de la imagen en píxeles
 * @param boxes   IntArray plano con las cajas NMS [x1,y1,x2,y2, x1,y1,x2,y2, ...]
 */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_sauczuk_cpp_1poc_1opencv_OpenCVCameraActivity_applySpotlightNative(
        JNIEnv* env,
        jobject /* this */,
        jintArray pixels,
        jint width,
        jint height,
        jintArray boxes) {

    const jint  pixelCount = width * height;
    const jint  boxLength  = env->GetArrayLength(boxes);
    const jint  boxCount   = boxLength / 4;

    jint* src     = env->GetIntArrayElements(pixels, nullptr);
    jint* boxData = env->GetIntArrayElements(boxes,  nullptr);

    jintArray result = env->NewIntArray(pixelCount);
    jint*     dst    = env->GetIntArrayElements(result, nullptr);

    // ── 1. Construir máscara: true = píxel dentro de alguna caja ──────────────
    std::vector<bool> mask(pixelCount, false);
    for (int b = 0; b < boxCount; ++b) {
        int x1 = std::max(0,         boxData[b * 4 + 0]);
        int y1 = std::max(0,         boxData[b * 4 + 1]);
        int x2 = std::min(width  - 1, boxData[b * 4 + 2]);
        int y2 = std::min(height - 1, boxData[b * 4 + 3]);
        for (int y = y1; y <= y2; ++y)
            for (int x = x1; x <= x2; ++x)
                mask[y * width + x] = true;
    }

    // ── 2. Aplicar spotlight: fondo oscuro, objetos detectados en brillo normal ─
    for (int i = 0; i < pixelCount; ++i) {
        jint pixel = src[i];
        int  a = (pixel >> 24) & 0xFF;
        int  r = (pixel >> 16) & 0xFF;
        int  g = (pixel >> 8)  & 0xFF;
        int  b =  pixel        & 0xFF;

        if (!mask[i]) {
            // Zona fuera de detecciones: oscurecer al 25%
            r = static_cast<int>(static_cast<float>(r) * 0.25f);
            g = static_cast<int>(static_cast<float>(g) * 0.25f);
            b = static_cast<int>(static_cast<float>(b) * 0.25f);
        }
        dst[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }

    // ── 3. Dibujar borde verde (#00FF00) de 3px alrededor de cada caja ─────────
    const jint GREEN  = (0xFF << 24) | (0x00 << 16) | (0xFF << 8) | 0x00;
    const int  BORDER = 3;

    for (int b = 0; b < boxCount; ++b) {
        int x1 = std::max(0,          boxData[b * 4 + 0]);
        int y1 = std::max(0,          boxData[b * 4 + 1]);
        int x2 = std::min(width  - 1, boxData[b * 4 + 2]);
        int y2 = std::min(height - 1, boxData[b * 4 + 3]);

        // Bordes superior e inferior
        for (int x = x1; x <= x2; ++x) {
            for (int t = 0; t < BORDER; ++t) {
                if (y1 + t < height) dst[(y1 + t) * width + x] = GREEN;
                if (y2 - t >= 0)     dst[(y2 - t) * width + x] = GREEN;
            }
        }
        // Bordes izquierdo y derecho
        for (int y = y1; y <= y2; ++y) {
            for (int t = 0; t < BORDER; ++t) {
                if (x1 + t < width) dst[y * width + (x1 + t)] = GREEN;
                if (x2 - t >= 0)    dst[y * width + (x2 - t)] = GREEN;
            }
        }
    }

    env->ReleaseIntArrayElements(pixels, src,     JNI_ABORT);
    env->ReleaseIntArrayElements(boxes,  boxData, JNI_ABORT);
    env->ReleaseIntArrayElements(result, dst,     0);

    return result;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_sauczuk_cpp_1poc_1opencv_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_sauczuk_cpp_1poc_1opencv_MainActivity_string2FromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string saludo = "Hola";
    return env->NewStringUTF(saludo.c_str());
}

/**
 * Procesamiento nativo de imagen desde CameraActivity.
 * Recibe píxeles RGBA (IntArray de Kotlin) y aplica conversión a escala de grises en C++.
 * Demuestra el pipeline: Kotlin → JNI → C++ → Kotlin
 */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_sauczuk_cpp_1poc_1opencv_CameraActivity_processFrameNative(
        JNIEnv* env,
        jobject /* this */,
        jintArray pixels,
        jint width,
        jint height) {

    jint count = width * height;
    jint* data = env->GetIntArrayElements(pixels, nullptr);

    jintArray result = env->NewIntArray(count);
    jint* out = env->GetIntArrayElements(result, nullptr);

    // Conversión RGBA → Escala de grises (fórmula luminancia ITU-R BT.601)
    for (int i = 0; i < count; i++) {
        jint pixel = data[i];
        int a = (pixel >> 24) & 0xFF;
        int r = (pixel >> 16) & 0xFF;
        int g = (pixel >> 8)  & 0xFF;
        int b =  pixel        & 0xFF;

        int gray = static_cast<int>(0.299f * static_cast<float>(r) + 0.587f * static_cast<float>(g) + 0.114f * static_cast<float>(b));

        out[i] = (a << 24) | (gray << 16) | (gray << 8) | gray;
    }

    env->ReleaseIntArrayElements(pixels, data, JNI_ABORT);
    env->ReleaseIntArrayElements(result, out, 0);

    return result;
}

