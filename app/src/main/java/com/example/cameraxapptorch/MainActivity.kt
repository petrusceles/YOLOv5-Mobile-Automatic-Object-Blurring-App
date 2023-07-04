package com.example.cameraxapptorch

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.ImageView
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
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var yolov5Detector: Yolov5Detector
    @SuppressLint("SimpleDateFormat")
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss")

    private var currentBitmap: Bitmap? = null
    private lateinit var sortTracker: Sort

    private var finalBoundingBoxes: MutableList<FloatArray> = mutableListOf()
    private var untrackedBoundingBoxes: MutableList<FloatArray> = mutableListOf()
    private var isCapture = false

    private var counter = 0

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
        yolov5Detector = Yolov5Detector(tfliteModel,options)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        val py = Python.getInstance()
        sortTracker = Sort(py.getModule("lsa"))


        viewBinding.captureButton.setOnClickListener {
            isCapture = !isCapture
            if (isCapture) {
                viewBinding.captureButton.text = "Stop Capture"
            } else {
                counter = 0
                viewBinding.captureButton.text = "Start Capture"
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

    private fun analyzer(imageProxy: ImageProxy) {
        val bitmap = ImageProcessor.imageProxyToBitmap(imageProxy,imageProxy.imageInfo.rotationDegrees)

        val startTime = SystemClock.uptimeMillis()

        var resizedBitmap = if (Yolov5Model.getGrayscale()) {
            ImageProcessor.scaleAndGrayScale(bitmap,yolov5Detector.inputShape[1],yolov5Detector.inputShape[2])
        } else if (Yolov5Model.getHisteq()) {
            ImageProcessor.scaleAndHisteq(
                bitmap,
                yolov5Detector.inputShape[1],
                yolov5Detector.inputShape[2]
            )
        } else {
            ImageProcessor.scaleOnly(
                bitmap,
                yolov5Detector.inputShape[1],
                yolov5Detector.inputShape[2])
        }
        yolov5Detector.createInputBuffer(resizedBitmap)

        executor.execute {
            currentBitmap = bitmap
            untrackedBoundingBoxes = yolov5Detector.inferenceAndPostProcess(bitmap.width,bitmap.height,resizedBitmap)

            for (box in untrackedBoundingBoxes) {
                Log.d("BOXES", box.contentToString())
            }

            finalBoundingBoxes = if (getIsTracking()) {
                sortTracker.updateSort(untrackedBoundingBoxes)
            } else {
                untrackedBoundingBoxes
            }

        }

        currentBitmap?.let { drawRectangleAndShow(it) }

        imageProxy.close()
        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
        Log.d("TIME SPENT", "$timeSpent ms")
    }


    fun drawResizedBitmapInImageView(resizedBitmap: Bitmap, imageView: ImageView) {
        val drawable = BitmapDrawable(imageView.resources, resizedBitmap)
        imageView.setImageDrawable(drawable)
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

    private fun drawRectangleAndShow(bitmap: Bitmap) {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val blurPaint = Paint().apply {
            maskFilter = BlurMaskFilter(10f, BlurMaskFilter.Blur.NORMAL)
        }
        for (box in finalBoundingBoxes) {
            val left = box[0].toInt()
            val top = box[1].toInt()
            val right = box[2].toInt()
            val bottom = box[3].toInt()
            val rect = Rect(left, top, right, bottom)
            val blurredRegion =
                Bitmap.createBitmap(rect.width(), rect.height(), Bitmap.Config.ARGB_8888)
            val blurredCanvas = Canvas(blurredRegion)
            blurredCanvas.drawBitmap(bitmap, -left.toFloat(), -top.toFloat(), null)
            val rs = RenderScript.create(this)
            val blurInput = Allocation.createFromBitmap(rs, blurredRegion)
            val blurOutput = Allocation.createTyped(rs, blurInput.type)
            val blurScript = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs))
            blurScript.setRadius(20f)
            blurScript.setInput(blurInput)
            blurScript.forEach(blurOutput)
            blurOutput.copyTo(blurredRegion)
            rs.destroy()
            canvas.drawBitmap(blurredRegion,left.toFloat(), top.toFloat(), blurPaint)
        }
        viewBinding.imageView.setImageBitmap(mutableBitmap)
        if (isCapture) {
            counter += 1
            val currentTime = dateFormat.format(Date()).replace(":", ".")
            val folder = File(getExternalFilesDir(null), "${Yolov5Model.getFolderMain()}-${Yolov5Model.getFolderPrefix()}")
            folder.mkdirs()
            val outputTxtFile = File(folder,"$currentTime-${counter.toString().padStart(4,'0')}.txt")
            val outputPngFile = File(folder,"$currentTime-${counter.toString().padStart(4,'0')}.jpg")
            saveBoundingBoxes(outputTxtFile, finalBoundingBoxes, bitmap.width, bitmap.height) // Save bounding box data
            saveImage(outputPngFile, bitmap) // Save image data
            if (Yolov5Model.getIsTracking() && Yolov5Model.getIsSaveUntracked()) {
                val outputTxtFileUntracked = File(folder, "$currentTime-${counter.toString().padStart(4,'0')}-untracked.txt")
                saveBoundingBoxes(outputTxtFileUntracked, untrackedBoundingBoxes, bitmap.width, bitmap.height)
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
}
