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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
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
    private val recorderExecutor = Executors.newSingleThreadExecutor()
    private lateinit var yolov5Detector: Yolov5Detector
    @SuppressLint("SimpleDateFormat")
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss")

    private var currentBitmap: Bitmap? = null
    private lateinit var mutableBitmap: Bitmap
    private lateinit var sortTracker: Sort

    private var finalBoundingBoxes: MutableList<FloatArray> = mutableListOf()
    private var untrackedBoundingBoxes: MutableList<FloatArray> = mutableListOf()
    private var isCapture = false
    private var isRecord = false


    private lateinit var mediaRecorder: MediaRecorder

    private var frameCounter = 0
    private var saveFolder: File? = null
    private var saveJob: Job? = null
    private var surface: Surface? = null

    private var captureCounter = 0

    private var textFileName: String? = null

    private var untrackedBoundingBoxesText = mutableListOf<MutableList<String>>()
    private var boundingBoxesText = mutableListOf<MutableList<String>>()


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
        val bitmap = ImageProcessor.imageProxyToBitmap(imageProxy,imageProxy.imageInfo.rotationDegrees)

        val startTime = SystemClock.uptimeMillis()

//        var resizedBitmap = ImageProcessor.resizeImageWithPadding(bitmap,512,512)
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
//        viewBinding.imageView.setImageBitmap(bitmap)
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

//        if (isCapture) {
//            saveData(bitmap, finalBoundingBoxes, untrackedBoundingBoxes)
//        }
        imageProxy.close()
        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
        Log.d("TIME SPENT", "$timeSpent ms")
    }

    @RequiresApi(Build.VERSION_CODES.S)
    private fun recordFrame(bitmap: Bitmap) {
        var startTime = SystemClock.uptimeMillis()
        try {
            // Start the recording if it's not already started
            if (!::mediaRecorder.isInitialized) {
                Log.d("RECORD", "START")
                mediaRecorder = MediaRecorder(this)
                mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC)
                mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE)
                mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)

                mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.HE_AAC)
                mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
                mediaRecorder.setAudioEncodingBitRate(64000)
                mediaRecorder.setVideoEncodingBitRate(10000000)
                mediaRecorder.setVideoFrameRate(24)

                mediaRecorder.setVideoSize(bitmap.width, bitmap.height)
                val videoOutputFile = getVideoOutputFile()
                textFileName = videoOutputFile.second
                mediaRecorder.setOutputFile(videoOutputFile.first)
                mediaRecorder.prepare()
                mediaRecorder.start()
                surface = mediaRecorder.surface
            }
            // Write the bitmap data to the media rechow order surface
            val canvas = surface?.lockCanvas(null)
            canvas?.drawBitmap(bitmap, 0f, 0f, null)
            surface?.unlockCanvasAndPost(canvas)


            if (saveFolder == null) {
                saveFolder = File(getExternalFilesDir(null), "${Yolov5Model.getFolderMain()}-${Yolov5Model.getFolderPrefix()}")
                saveFolder!!.mkdirs()
            }

            if (Yolov5Model.getIsSaveUntracked()) {
                val boundingBoxesTextValues = collectBoundingBoxesOnRecord(finalBoundingBoxes,bitmap.width, bitmap.height)
                boundingBoxesText.add(boundingBoxesTextValues)
                if (Yolov5Model.getIsTracking()) {
                    val untrackedBoundingBoxesTextValues = collectBoundingBoxesOnRecord(untrackedBoundingBoxes,bitmap.width, bitmap.height)
                    untrackedBoundingBoxesText.add(untrackedBoundingBoxesTextValues)
                }
            }


        } catch (e: Exception) {
            Log.e(TAG, "Failed to record frame", e)
        }

        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
        Log.d("TIME SPENT RECORD", "$timeSpent ms")
    }

    private fun stopRecording() {
        try {
            // Stop and release the media recorder
            mediaRecorder.stop()
            mediaRecorder.reset()
            mediaRecorder.release()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop recording", e)
        }
        var captureCounter = 0
        var captureCounterUntracked = 0
        if (Yolov5Model.getIsSaveUntracked()) {
            for (boundingBoxesFrame in boundingBoxesText) {
                saveBoundingBoxesTextOnRecord(File(saveFolder, "${textFileName}-${captureCounter.toString().padStart(4, '0')}.txt"), boundingBoxesFrame)
                captureCounter++
            }
            if (Yolov5Model.getIsTracking()) {
                for (untrackedBoundingBoxesFrame in untrackedBoundingBoxesText) {
                    saveBoundingBoxesTextOnRecord(File(saveFolder, "${textFileName}-${captureCounterUntracked.toString().padStart(4, '0')}-untracked.txt"), untrackedBoundingBoxesFrame)
                    captureCounterUntracked++
                }
            }
        }
        showToast("DONE")
        boundingBoxesText.clear()
        untrackedBoundingBoxesText.clear()
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

    private fun saveBoundingBoxesOnRecord(file: File, boundingBoxes: List<FloatArray>, imageWidth: Int, imageHeight: Int,frame: Int) {
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

                val boundingBoxText = "$frame $center_x $center_y $width $height\n"
                fileOutputStream.write(boundingBoxText.toByteArray())
            }
            fileOutputStream.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error saving bounding box data: ${e.message}")
        }
    }

    private fun saveBoundingBoxesTextOnRecord(file: File, boundingBoxes: List<String>) {
        try {
            val fileOutputStream = FileOutputStream(file)
            for (box in boundingBoxes) {
                fileOutputStream.write(box.toByteArray())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error saving bounding box data: ${e.message}")
        }
    }

    private fun collectBoundingBoxesOnRecord(boundingBoxes: List<FloatArray>, imageWidth: Int, imageHeight: Int): MutableList<String> {
        val boundingBoxesText = mutableListOf<String>()
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
            boundingBoxesText.add(boundingBoxText)
        }
        return boundingBoxesText
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
        // Increment frame counter
        frameCounter++

        // Create save folder if it doesn't exist
        if (saveFolder == null) {
            saveFolder = File(getExternalFilesDir(null), "${Yolov5Model.getFolderMain()}-${Yolov5Model.getFolderPrefix()}")
            saveFolder!!.mkdirs()
        }

        // Cancel previous save job if it's still running
        saveJob?.cancel()

        // Start a new save job
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
    // Call this method when recording is stopped or paused to cancel any ongoing save job
    private fun cancelSaveJob() {
        saveJob?.cancel()
        saveJob = null
    }

    @RequiresApi(Build.VERSION_CODES.S)
    private fun drawRectangleAndShow(bitmap: Bitmap) {
        mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val blurPaint = Paint().apply {
            maskFilter = BlurMaskFilter(10f, BlurMaskFilter.Blur.NORMAL)
        }
        val rectPaint = Paint().apply {
            color = Color.GREEN
            strokeWidth = 2.0f
            style = Paint.Style.STROKE
        }
        for (box in finalBoundingBoxes) {
            Log.d("BOX", box.contentToString())
            val left = box[0].toInt()
            val top = box[1].toInt()
            val right = box[2].toInt()
            val bottom = box[3].toInt()
            val rect = Rect(left, top, right, bottom)
            if (rect.width() <= 0 || rect.height() <= 0) {
                continue
            }
            canvas.drawRect(rect,rectPaint)

//            val blurredRegion =
//                Bitmap.createBitmap(rect.width(), rect.height(), Bitmap.Config.ARGB_8888)
//            val blurredCanvas = Canvas(blurredRegion)
//            blurredCanvas.drawBitmap(bitmap, -left.toFloat(), -top.toFloat(), null)
//            val rs = RenderScript.create(this)
//            val blurInput = Allocation.createFromBitmap(rs, blurredRegion)
//            val blurOutput = Allocation.createTyped(rs, blurInput.type)
//            val blurScript = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs))
//            blurScript.setRadius(20f)
//            blurScript.setInput(blurInput)
//            blurScript.forEach(blurOutput)
//            blurOutput.copyTo(blurredRegion)
//            rs.destroy()
//            canvas.drawBitmap(blurredRegion,left.toFloat(), top.toFloat(), blurPaint)
        }
        viewBinding.imageView.setImageBitmap(mutableBitmap)
        if (isRecord) {
            if (Yolov5Model.getIsSaveUntracked()) {
                recordFrame(bitmap)
            } else {
                recordFrame(mutableBitmap)
            }
        }
        if (isCapture) {
            saveData(bitmap, finalBoundingBoxes, untrackedBoundingBoxes)
        }
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