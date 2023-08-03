package com.example.cameraxapptorch

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.media.EncoderProfiles
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore.Audio.Media
import android.util.Log
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.ImageView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.CameraSelector.LENS_FACING_BACK
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.renderscript.Allocation
import androidx.renderscript.Element
import androidx.renderscript.RenderScript
import androidx.renderscript.ScriptIntrinsicBlur
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.cameraxapptorch.Yolov5Model.IMAGE_MEAN
import com.example.cameraxapptorch.Yolov5Model.IMAGE_STD
import com.example.cameraxapptorch.Yolov5Model.getIsTracking
import com.example.cameraxapptorch.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.apache.commons.math3.linear.*
import org.apache.commons.math3.optim.linear.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var yolov5Detector: Yolov5Detector
    @SuppressLint("SimpleDateFormat")
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss")

    private var isCapture = false
    private var isRecord = false

    private lateinit var mediaRecorder: MediaRecorder

    private var frameCounter = 0
    private var saveFolder: File? = null
    private var saveJob: Job? = null
    private var surface: Surface? = null

    private var startShowing = false

    private val scope = CoroutineScope(Dispatchers.Default)

    private var textFileName: String? = null
    private lateinit var originalBitmap: Bitmap
    private lateinit var originalBitmapBlock2: Bitmap
    private lateinit var originalBitmapBlock3: Bitmap

    @RequiresApi(Build.VERSION_CODES.S)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        val windowInsetsController =
            WindowCompat.getInsetsController(window, window.decorView)
        windowInsetsController.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        windowInsetsController.hide(WindowInsetsCompat.Type.systemBars())

        val tfliteModel: MappedByteBuffer = Yolov5Model.getMappedByteBuffer()
        val options = Interpreter.Options()

        options.useNNAPI = true
        options.numThreads = 2
        yolov5Detector = Yolov5Detector(tfliteModel,options, this)


        viewBinding.captureButton.setOnClickListener {
            isCapture = !isCapture
            if (isCapture) {
                viewBinding.captureButton.text = "Stop Capture"
            } else {
                cancelSaveJob()
                viewBinding.captureButton.text = "Start Capture"
            }
        }

        viewBinding.recordButton.setOnClickListener {
            isRecord = !isRecord
            if (isRecord) {
                viewBinding.recordButton.text = "Stop Record"
            } else {
                viewBinding.recordButton.text = "Start Record"
                stopRecording()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }

    @RequiresApi(Build.VERSION_CODES.S)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
            imageAnalyzer.setAnalyzer(
                ContextCompat.getMainExecutor(this)
            ) { image: ImageProxy -> analyzer(image) }
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(LENS_FACING_BACK).build()
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector,  imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }


    @RequiresApi(Build.VERSION_CODES.S)
    private fun analyzer(imageProxy: ImageProxy) {
//        val startTime = SystemClock.uptimeMillis()
//        originalBitmap = ImageProcessor.imageProxyToBitmap(imageProxy, imageProxy.imageInfo.rotationDegrees)
//
//        Log.d("BITMAP INITIALIZATION", "ORIGINAL " + ::originalBitmap.isInitialized.toString())
//        if (::originalBitmap.isInitialized) {
//            originalBitmapBlock2 = originalBitmap.copy(originalBitmap.config, true)
//            val resizedBitmap = if (Yolov5Model.getGrayscale()) {
//                ImageProcessor.resizeAndGrayScale(
//                    originalBitmapBlock2,
//                    yolov5Detector.inputShape[1],
//                    yolov5Detector.inputShape[2]
//                )
//            } else if (Yolov5Model.getHisteq()) {
//                ImageProcessor.resizeAndHisteq(
//                    originalBitmapBlock2,
//                    yolov5Detector.inputShape[1],
//                    yolov5Detector.inputShape[2]
//                )
//            } else {
//                ImageProcessor.resizeOnly(
//                    originalBitmapBlock2,
//                    yolov5Detector.inputShape[1],
//                    yolov5Detector.inputShape[2]
//                )
//            }
//            yolov5Detector.createInputBuffer(resizedBitmap)
//            originalBitmapBlock3 = originalBitmapBlock2.copy(originalBitmapBlock2.config,true)
//        }
//
//
//        Log.d("BITMAP INITIALIZATION", "CURRENT " + yolov5Detector.isCurrentBitmapInitialized().toString())
//        if (::originalBitmapBlock3.isInitialized) {
//            yolov5Detector.inferenceAndPostProcess(originalBitmapBlock3)
//            startShowing = true
//        }
//
//        imageProxy.close()
//
//        if (startShowing) {
//            if (isRecord) {
//                recordFrame(yolov5Detector.getMutableBitmap())
//            }
//            if (isCapture) {
//                saveData(
//                    yolov5Detector.getCurrentBitmap(),
//                    yolov5Detector.getFinalBoundingBoxes(),
//                    yolov5Detector.getUntrackedBoundingBoxes()
//                )
//            }
//            viewBinding.imageView.setImageBitmap(yolov5Detector.getMutableBitmap())
//        }
//
//        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
//        Log.d("TIME SPENT", "$timeSpent ms")


        scope.launch {
            val startTime = SystemClock.uptimeMillis()
            val block0 = async {
                originalBitmap = ImageProcessor.imageProxyToBitmap(imageProxy, imageProxy.imageInfo.rotationDegrees)
            }
            val block1 = async {
                Log.d("BITMAP INITIALIZATION", "ORIGINAL " + ::originalBitmap.isInitialized.toString())
                if (::originalBitmap.isInitialized) {
                    originalBitmapBlock2 = originalBitmap.copy(originalBitmap.config, true)
                    val resizedBitmap = if (Yolov5Model.getGrayscale()) {
                        ImageProcessor.resizeAndGrayScale(
                            originalBitmapBlock2,
                            yolov5Detector.inputShape[1],
                            yolov5Detector.inputShape[2]
                        )
                    } else if (Yolov5Model.getHisteq()) {
                        ImageProcessor.resizeAndHisteq(
                            originalBitmapBlock2,
                            yolov5Detector.inputShape[1],
                            yolov5Detector.inputShape[2]
                        )
                    } else {
                        ImageProcessor.resizeOnly(
                            originalBitmapBlock2,
                            yolov5Detector.inputShape[1],
                            yolov5Detector.inputShape[2]
                        )
                    }
                    yolov5Detector.createInputBuffer(resizedBitmap)
                    originalBitmapBlock3 = originalBitmapBlock2.copy(originalBitmapBlock2.config,true)
                }
            }
            val block2 = async {
                Log.d("BITMAP INITIALIZATION", "CURRENT " + yolov5Detector.isCurrentBitmapInitialized().toString())
                if (::originalBitmapBlock3.isInitialized) {
                    yolov5Detector.inferenceAndPostProcess(originalBitmapBlock3)
                    startShowing = true
                }
            }

//            block0.await()
//            block1.await()
//            block2.await()
            awaitAll(block0,block1, block2)

            imageProxy.close()
            if (startShowing) {
                if (isRecord) {
                    recordFrame(yolov5Detector.getMutableBitmap())
                }
                if (isCapture) {
                    saveData(
                        yolov5Detector.getCurrentBitmap(),
                        yolov5Detector.getFinalBoundingBoxes(),
                        yolov5Detector.getUntrackedBoundingBoxes()
                    )
                }
                withContext(Dispatchers.Main) {
                    viewBinding.imageView.setImageBitmap(yolov5Detector.getMutableBitmap())
                }
            }
                val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
                Log.d("TIME SPENT", "$timeSpent ms")
        }


//        originalBitmap = ImageProcessor.imageProxyToBitmap(imageProxy, imageProxy.imageInfo.rotationDegrees)
//
//        viewBinding.imageView.setImageBitmap(originalBitmap)
//        if (isRecord) {
//            recordFrame(originalBitmap)
//            val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
//            Log.d("TIME SPENT", "$timeSpent ms")
//        }
//        imageProxy.close()
    }

    @RequiresApi(Build.VERSION_CODES.S)
    private fun recordFrame(bitmap: Bitmap) {
//        var startTime = SystemClock.uptimeMillis()
        try {
            // Start the recording if it's not already started
            if (!::mediaRecorder.isInitialized) {
                Log.d("RECORD", "START")
                mediaRecorder = MediaRecorder(this)
                mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC)
                mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE)
                mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)

                mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
                mediaRecorder.setAudioEncodingBitRate(64000)
                mediaRecorder.setVideoEncodingBitRate(1500000)
                mediaRecorder.setVideoFrameRate(240)
                mediaRecorder.setVideoSize(bitmap.width, bitmap.height)
                val videoOutputFile = getVideoOutputFile()
                textFileName = videoOutputFile.second
                mediaRecorder.setOutputFile(videoOutputFile.first)
                mediaRecorder.prepare()
                mediaRecorder.start()
                surface = mediaRecorder.surface
            }
            val canvas = surface?.lockCanvas(null)
            canvas?.drawBitmap(bitmap, 0f, 0f, null)
            surface?.unlockCanvasAndPost(canvas)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to record frame", e)
        }

//        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
//        Log.d("TIME SPENT RECORD", "$timeSpent ms")
    }

    private fun stopRecording() {
        try {
            mediaRecorder.stop()
            mediaRecorder.reset()
            mediaRecorder.release()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop recording", e)
        }
        showToast("DONE")
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun saveBoundingBoxes(file: File, boundingBoxes: List<FloatArray>, imageWidth: Int, imageHeight: Int) {
        try {
            val fileOutputStream = FileOutputStream(file)
            for (box in boundingBoxes) {
                val x1 = box[0]
                val y1 = box[1]
                val x2 = box[2]
                val y2 = box[3]

                val center_x = ((x1 + x2) / 2)/imageWidth
                val center_y = ((y1 + y2) / 2)/imageHeight
                val width = (x2 - x1)/imageWidth
                val height = (y2 - y1)/imageHeight

                val boundingBoxText = "0 $center_x $center_y $width $height\n"
                fileOutputStream.write(boundingBoxText.toByteArray())
            }
            fileOutputStream.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error saving bounding box data: ${e.message}")
        }
    }

    private fun saveImage(file: File, bitmap: Bitmap) {
        try {
            val fileOutputStream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
            fileOutputStream.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error saving image data: ${e.message}")
        }
    }

    // Call this method to save the tracked bounding box, untracked bounding box, and image
    private fun saveData(bitmap: Bitmap, finalBoundingBoxes: MutableList<FloatArray>, untrackedBoundingBoxes: MutableList<FloatArray>) {
        frameCounter++

        if (saveFolder == null) {
            saveFolder = File(getExternalFilesDir(null), "${Yolov5Model.getFolderMain()}-${Yolov5Model.getFolderPrefix()}")
            saveFolder!!.mkdirs()
        }

        saveJob?.cancel()

        saveJob = GlobalScope.launch(Dispatchers.IO) {
            val currentTime = dateFormat.format(Date()).replace(":", ".")
            val counter = frameCounter

            // Save tracked bounding box data
            val outputTxtFile = File(saveFolder, "$currentTime-${counter.toString().padStart(4, '0')}.txt")
            saveBoundingBoxes(outputTxtFile, finalBoundingBoxes, bitmap.width, bitmap.height)

            // Save image data
            val outputPngFile = File(saveFolder, "$currentTime-${counter.toString().padStart(4, '0')}.jpg")
            saveImage(outputPngFile, bitmap)

            // Save untracked bounding box data (if needed)
            if (Yolov5Model.getIsTracking() && Yolov5Model.getIsSaveUntracked()) {
                val outputTxtFileUntracked = File(saveFolder, "$currentTime-${counter.toString().padStart(4, '0')}-untracked.txt")
                saveBoundingBoxes(outputTxtFileUntracked, untrackedBoundingBoxes, bitmap.width, bitmap.height)
            }
        }
    }

    private fun cancelSaveJob() {
        saveJob?.cancel()
        saveJob = null
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun getVideoOutputFile() : Pair<File,String> {
        val currentTime = dateFormat.format(Date()).replace(":", ".")
        val folder = File(getExternalFilesDir(null), "${Yolov5Model.getFolderMain()}-${Yolov5Model.getFolderPrefix()}-video")
        folder.mkdirs()
        val outputMp4File = File(folder,"$currentTime.mp4")
        return Pair(outputMp4File,currentTime)
    }
}