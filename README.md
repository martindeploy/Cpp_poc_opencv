# Cpp POC OpenCV — Android con Kotlin + C++ + OpenCV

Proof of Concept que demuestra la integración de **OpenCV**, **CameraX** y código **C++ nativo (JNI)** en una app Android escrita en Kotlin.

---

## Tabla de contenidos

1. [Estructura del proyecto](#1-estructura-del-proyecto)
2. [Cómo Kotlin se vincula con C++ — JNI](#2-cómo-kotlin-se-vincula-con-c-jni)
3. [Cómo se integra OpenCV](#3-cómo-se-integra-opencv)
4. [Activity 1 — Cámara con Canny Edge Detection](#4-activity-1--cámara-con-canny-edge-detection)
5. [Activity 2 — YOLO con OpenCV DNN + C++ Spotlight](#5-activity-2--yolo-con-opencv-dnn--c-spotlight)
6. [Por qué no hay #include opencv en C++](#6-por-qué-no-hay-include-opencv-en-c)
7. [Cómo habilitar OpenCV en C++](#7-cómo-habilitar-opencv-en-c)
8. [Setup del modelo YOLO](#8-setup-del-modelo-yolo)
9. [Dependencias](#9-dependencias)

---

## 1. Estructura del proyecto

```
app/src/main/
├── AndroidManifest.xml             # Permisos + registro de activities
├── assets/
│   ├── coco.names                  # 80 clases del dataset COCO
│   ├── yolov4-tiny.cfg             # Arquitectura de la red YOLO
│   └── yolov4-tiny.weights         # ⚠️ Descargar manualmente (ver §8)
├── cpp/
│   ├── CMakeLists.txt              # Build de la librería nativa
│   └── native-lib.cpp              # Funciones JNI en C++
├── java/com/sauczuk/cpp_poc_opencv/
│   ├── MainActivity.kt             # Pantalla principal con 2 botones
│   ├── CameraActivity.kt           # Cámara + OpenCV Canny (Java API)
│   └── OpenCVCameraActivity.kt     # Cámara + YOLO + C++ Spotlight
└── res/layout/
    ├── activity_main.xml
    ├── activity_camera.xml
    └── activity_opencv_camera.xml
```

---

## 2. Cómo Kotlin se vincula con C++ — JNI

El puente entre Kotlin y C++ se llama **JNI (Java Native Interface)**. Hay 4 piezas que trabajan juntas:

### Pieza 1 — `build.gradle.kts` le dice a Gradle que compile C++

```kotlin
externalNativeBuild {
    cmake {
        path = file("src/main/cpp/CMakeLists.txt")
    }
}
```

### Pieza 2 — `CMakeLists.txt` compila el C++ como librería `.so`

```cmake
add_library(cpp_poc_opencv SHARED native-lib.cpp)
target_link_libraries(cpp_poc_opencv android log)
```

Produce `libcpp_poc_opencv.so` para cada arquitectura (arm64-v8a, x86_64, etc.).

### Pieza 3 — Kotlin declara y carga la librería

```kotlin
// Declara que la función está implementada en C++
external fun stringFromJNI(): String

companion object {
    init {
        System.loadLibrary("cpp_poc_opencv")  // carga el .so en tiempo de ejecución
    }
}
```

La palabra clave `external` le dice a Kotlin: *"esta función no está aquí, búscala en la librería nativa"*.

### Pieza 4 — C++ implementa la función con la firma correcta

```cpp
extern "C" JNIEXPORT jstring JNICALL
Java_com_sauczuk_cpp_1poc_1opencv_MainActivity_stringFromJNI(
        JNIEnv* env, jobject /* this */) {
    return env->NewStringUTF("Hello from C++");
}
```

El nombre de la función sigue una **convención estricta**:

```
Java_ | com_sauczuk_cpp_1poc_1opencv | MainActivity | stringFromJNI
  ↑              ↑                         ↑               ↑
prefijo       paquete                    clase          método Kotlin
             (. → _)
             (_ → _1)
```

> **Regla importante**: los guiones bajos `_` en el paquete o clase se escapan como `_1` en JNI.
> El paquete `com.sauczuk.cpp_poc_opencv` se convierte en `com_sauczuk_cpp_1poc_1opencv`.

### Flujo completo

```
MainActivity.kt
   │
   ├─ System.loadLibrary("cpp_poc_opencv")  →  carga libcpp_poc_opencv.so
   │
   └─ stringFromJNI()  ──JNI──▶  Java_com_sauczuk_cpp_1poc_1opencv_MainActivity_stringFromJNI()
                                          en native-lib.cpp
```

---

## 3. Cómo se integra OpenCV

OpenCV se agrega como dependencia Maven en `libs.versions.toml`:

```toml
[versions]
opencv = "4.11.0"

[libraries]
opencv = { group = "org.opencv", name = "opencv", version.ref = "opencv" }
```

Y en `app/build.gradle.kts`:

```kotlin
implementation(libs.opencv)
```

Este AAR proporciona:
- ✅ La API Java/Kotlin completa (`org.opencv.core.*`, `org.opencv.imgproc.*`, `org.opencv.dnn.*`, etc.)
- ✅ La librería nativa precompilada `libopencv_java4.so`
- ❌ **No** expone los headers C++ para compilación NDK (ver §6)

OpenCV se inicializa al arrancar cada Activity:

```kotlin
if (!OpenCVLoader.initLocal()) {
    // error
}
```

---

## 4. Activity 1 — Cámara con Canny Edge Detection

**`CameraActivity.kt`** — Pantalla dividida en dos mitades:
- **Superior**: preview en vivo de la cámara (CameraX)
- **Inferior**: resultado del filtro Canny en tiempo real (OpenCV)

### Pipeline

```
CameraX (ImageAnalysis)
    │
    ▼ frame RGBA
Bitmap → Mat  (Utils.bitmapToMat)
    │
    ▼
cvtColor RGBA→GRAY  (OpenCV Java)
    │
    ▼
Canny(80, 180)      (OpenCV Java — detección de bordes)
    │
    ▼
Mat → Bitmap  (Utils.matToBitmap)
    │
    ▼  también pasa por C++:
processFrameNative()  ──JNI──▶  conversión RGBA→gris en C++
    │
    ▼
ImageView (resultado Canny)
```

### Función JNI en C++ — `processFrameNative`

Recibe píxeles RGBA como `IntArray`, aplica la fórmula de luminancia ITU-R BT.601 en C++ y devuelve el resultado en escala de grises:

```cpp
int gray = static_cast<int>(
    0.299f * r + 0.587f * g + 0.114f * b
);
out[i] = (a << 24) | (gray << 16) | (gray << 8) | gray;
```

Demuestra el pipeline completo: **Kotlin → JNI → C++ → Kotlin**.

---

## 5. Activity 2 — YOLO con OpenCV DNN + C++ Spotlight

**`OpenCVCameraActivity.kt`** — Pantalla dividida en dos mitades:
- **Superior**: preview en vivo de la cámara (CameraX)
- **Inferior**: detección de objetos YOLO con efecto spotlight

### Pipeline completo

```
CameraX (ImageAnalysis)
    │
    ▼ Bitmap RGBA
1. blobFromImage(416×416)     ← preprocesar frame para la red
    │
    ▼
2. net.forward()              ← inferencia YOLOv4-tiny en CPU
    │                            (OpenCV DNN Java API)
    ▼
3. Parsear detecciones        ← [cx, cy, w, h, objectness, clases...]
    │
    ▼
4. NMSBoxes()                 ← Non-Maximum Suppression: eliminar duplicados
    │
    ▼ cajas finales [x1,y1,x2,y2]
5. applySpotlightNative() ──JNI──▶  C++ procesa píxeles:
    │                                  · oscurece fondo al 25%
    │                                  · mantiene regiones detectadas
    │                                  · dibuja bordes verdes de 3px
    ▼
6. Imgproc.putText()          ← etiquetas de clase (OpenCV Java)
    │
    ▼
ImageView (resultado final)
```

### Función JNI en C++ — `applySpotlightNative`

Recibe los píxeles del frame y las cajas de detección ya filtradas por NMS. Aplica tres operaciones en C++:

**1. Construye una máscara** — marca qué píxeles están dentro de alguna caja:
```cpp
std::vector<bool> mask(pixelCount, false);
for (int b = 0; b < boxCount; ++b) {
    for (int y = y1; y <= y2; ++y)
        for (int x = x1; x <= x2; ++x)
            mask[y * width + x] = true;
}
```

**2. Aplica el efecto spotlight** — oscurece el fondo, mantiene objetos:
```cpp
if (!mask[i]) {
    r = static_cast<int>(static_cast<float>(r) * 0.25f);
    g = static_cast<int>(static_cast<float>(g) * 0.25f);
    b = static_cast<int>(static_cast<float>(b) * 0.25f);
}
```

**3. Dibuja bordes verdes de 3px** alrededor de cada caja:
```cpp
const jint GREEN = (0xFF << 24) | (0x00 << 16) | (0xFF << 8) | 0x00;
// dibuja top, bottom, left, right con 3px de grosor
```

### Carga del modelo

El modelo se carga desde `assets/` en un hilo de fondo para no bloquear la UI. Los archivos se copian al almacenamiento interno (porque `Dnn.readNetFromDarknet` necesita rutas de archivo, no streams):

```kotlin
val cfg     = copyAssetToFile("yolov4-tiny.cfg")
val weights = copyAssetToFile("yolov4-tiny.weights")
net = Dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(Dnn.DNN_TARGET_CPU)
```

Si el archivo `.weights` no existe, la Activity muestra la cámara en vivo con un aviso.

---

## 6. Por qué no hay `#include <opencv2/...>` en C++

El archivo `native-lib.cpp` **no incluye headers de OpenCV** aunque la librería está integrada en el proyecto. Esto es por cómo se obtiene OpenCV:

```
org.opencv:opencv:4.11.0  (Maven AAR)
        │
        ├── Java/Kotlin API  ✅  → org.opencv.dnn.Dnn, Imgproc, Core...
        │                            (usado en CameraActivity y OpenCVCameraActivity)
        │
        └── Headers C++ (.hpp) ❌  → NO expuestos automáticamente al compilador NDK
```

El AAR de Maven empaqueta la librería `.so` precompilada y los wrappers Java, pero **no registra los headers `.hpp` en el classpath del build NDK/CMake**. Por eso las funciones C++ actuales usan manipulación directa de píxeles en lugar de `cv::Mat`, `cv::rectangle`, etc.

| | Qué hace | Dónde corre |
|---|---|---|
| `OpenCV Java API` | YOLO DNN, Canny, putText, bitmapToMat | Kotlin/JVM |
| `Funciones JNI` | Spotlight, escala de grises | C++ puro (sin headers OpenCV) |

---

## 7. Cómo habilitar OpenCV en C++

Existen dos opciones para poder usar `#include <opencv2/opencv.hpp>` en el código C++:

### Opción A — Prefab (sin descargar nada extra, desde OpenCV 4.7+)

En `app/build.gradle.kts`, activar prefab:
```kotlin
buildFeatures {
    viewBinding = true
    prefab = true          // ← activa la exposición de headers del AAR
}
```

En `CMakeLists.txt`:
```cmake
find_package(OpenCV REQUIRED CONFIG)

target_link_libraries(${CMAKE_PROJECT_NAME}
    opencv_java4
    android
    log)
```

Luego en C++:
```cpp
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// Ahora funciona:
cv::Mat frame;
cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg, weights);
cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,255,0), 2);
```

> ⚠️ El soporte prefab del AAR oficial puede ser inconsistente dependiendo de la versión de OpenCV y AGP. Es la opción más rápida si funciona.

### Opción B — SDK completo (garantizado al 100%)

1. Descargar [OpenCV Android SDK](https://opencv.org/releases/) y descomprimir, por ejemplo en `app/src/main/cpp/opencv/`

2. En `CMakeLists.txt`:
```cmake
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni")
find_package(OpenCV REQUIRED)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OpenCV_LIBS}
    android
    log)
```

3. En C++:
```cpp
#include <opencv2/opencv.hpp>    // funciona con todo
#include <opencv2/dnn.hpp>       // módulo DNN para YOLO en C++
```

| | Maven AAR (actual) | Prefab | SDK completo |
|---|---|---|---|
| `#include <opencv2/...>` en C++ | ❌ | ✅ (puede variar) | ✅ |
| OpenCV en Kotlin/Java | ✅ | ✅ | ✅ |
| Requiere descarga extra | No | No | Sí (~200 MB) |
| Garantía de compilación | ✅ | ⚠️ | ✅ |

---

## 8. Setup del modelo YOLO

El archivo de pesos **no está incluido** en el repositorio por su tamaño (~23 MB). Hay que descargarlo manualmente:

```bash
# Descargar yolov4-tiny.weights y colocarlo en assets/
curl -L \
  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights \
  -o app/src/main/assets/yolov4-tiny.weights
```

Los demás archivos ya están incluidos en `assets/`:

| Archivo | Tamaño | Descripción |
|---|---|---|
| `yolov4-tiny.cfg` | ~4 KB | Arquitectura de la red (Darknet) |
| `coco.names` | ~1 KB | 80 nombres de clases COCO |
| `yolov4-tiny.weights` | ~23 MB | ⚠️ **Descargar manualmente** |

### Clases detectables (COCO)

La red puede detectar 80 categorías: `person`, `bicycle`, `car`, `dog`, `cat`, `bottle`, `chair`, `laptop`, `cell phone`, `book`, entre otras.

### Parámetros de detección

```kotlin
const val CONF_THRESHOLD = 0.45f   // confianza mínima para aceptar una detección
const val NMS_THRESHOLD  = 0.4f    // umbral para Non-Maximum Suppression
const val INPUT_SIZE     = 416.0   // resolución de entrada de YOLOv4-tiny
```

---

## 9. Dependencias

```toml
# gradle/libs.versions.toml

[versions]
opencv       = "4.11.0"
cameraX      = "1.4.1"
appcompat    = "1.6.1"
material     = "1.10.0"

[libraries]
opencv              = { group = "org.opencv",          name = "opencv",          version.ref = "opencv"  }
camerax-core        = { group = "androidx.camera",     name = "camera-core",     version.ref = "cameraX" }
camerax-camera2     = { group = "androidx.camera",     name = "camera-camera2",  version.ref = "cameraX" }
camerax-lifecycle   = { group = "androidx.camera",     name = "camera-lifecycle",version.ref = "cameraX" }
camerax-view        = { group = "androidx.camera",     name = "camera-view",     version.ref = "cameraX" }
```

---

## Notas adicionales

- **minSdk**: 24 (Android 7.0). CameraX y OpenCV 4.11 lo requieren como mínimo.
- **Orientación**: ambas Activities de cámara están bloqueadas en `portrait`.
- **Rendimiento YOLO**: se ejecuta en CPU. En dispositivos de gama baja puede ir a 1-3 FPS. Para producción, considerar `DNN_TARGET_VULKAN` o un modelo ONNX más pequeño (YOLOv8n).
- **Memory leaks**: todos los `Mat` se liberan con `.release()` después de su uso para evitar fugas de memoria nativa.
- **Frames saltados**: `OpenCVCameraActivity` usa `AtomicBoolean isProcessing` para descartar frames mientras el anterior se sigue procesando, evitando saturar el hilo.

