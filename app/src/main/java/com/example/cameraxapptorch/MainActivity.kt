package com.example.cameraxapptorch

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.core.CameraSelector.LENS_FACING_BACK
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.VideoCapture
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.PermissionChecker
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.renderscript.Allocation
import androidx.renderscript.Element
import androidx.renderscript.RenderScript
import androidx.renderscript.ScriptIntrinsicBlur
import com.example.cameraxapptorch.databinding.ActivityMainBinding
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.optim.linear.*
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import org.apache.commons.math3.optim.linear.*
import org.apache.commons.math3.linear.*
import org.apache.commons.math3.optimization.linear.SimplexSolver
import org.apache.commons.math3.optimization.linear.UnboundedSolutionException
import com.chaquo.python.PyException
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.io.IOException


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null

    private lateinit var cameraExecutor: ExecutorService

    private val executor = Executors.newSingleThreadExecutor()

    private lateinit var interpreter: Interpreter
    private lateinit var inputTensor: Tensor
    private lateinit var inputShape: IntArray
    private lateinit var inputType: DataType
    private var inputSize: Int = 0
    private lateinit var outputTensor: Tensor
    private lateinit var outputShape: IntArray
    private lateinit var outputType: DataType
    private var outputSize: Int = 0

    //SORT
    private var objectTrackers: MutableList<KalmanBoxTracker> = mutableListOf()
    private val maxAge: Int = 1
    private val minHits: Int = 3
    private val iouThreshold: Float = 0.3f
    private var frameCount: Int = 0
    private lateinit var module: PyObject

    private var finalBoundingBoxes: MutableList<FloatArray> = mutableListOf()

    private var isRecording = false
    private lateinit var mediaRecorder: MediaRecorder

    private var finalBitmap: Bitmap? = null

    private lateinit var surfaceHolder: SurfaceHolder
    private lateinit var surfaceView: SurfaceView

    private val dequantizeFactor = 0.013170517981052399
    private val dequantizeBias = 3

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        val windowInsetsController =
            WindowCompat.getInsetsController(window, window.decorView)
        // Configure the behavior of the hidden system bars.
        windowInsetsController.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        windowInsetsController.hide(WindowInsetsCompat.Type.systemBars())
        val tfliteModel = loadModelFile(this)
        val options = Interpreter.Options()
        options.useNNAPI = true
        options.numThreads = 2
        interpreter = Interpreter(tfliteModel, options)

        inputTensor = interpreter.getInputTensor(0)
        inputShape = inputTensor.shape()  // [batch_size, height, width, channels]
        inputType = inputTensor.dataType()

        inputSize = inputShape[1] * inputShape[2] * inputShape[3]  // height * width * channels


        outputTensor = interpreter.getOutputTensor(0)
        outputShape = outputTensor.shape()  // [batch_size, height, width, channels]
        outputType = outputTensor.dataType()

        outputSize = outputShape[0] * outputShape[1] * outputShape[2]  // height * width * channels
        interpreter = Interpreter(tfliteModel, options)


        // Set up the listeners for take photo and video capture buttons
//        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        val py = Python.getInstance()
        this.module = py.getModule("lsa")

        viewBinding.videoCaptureButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecording()
            }
        }
        surfaceView = viewBinding.surfaceView
        surfaceHolder = surfaceView.holder
        surfaceHolder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                // Surface created, you can now draw on it
                // Display your bitmap here
                drawBitmapOnSurface()
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                // Surface destroyed, clean up resources
                // Surface size or format
                drawBitmapOnSurface()
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
            }
        })

    }


    private fun drawBitmapOnSurface() {
        val canvas = surfaceHolder.lockCanvas()
        try {
            if (canvas != null) {
                // Clear the canvas
                canvas.drawColor(Color.BLACK)

                // Calculate the scaling factors for the bitmap to fit the screen
                val surfaceWidth = surfaceView.width.toFloat()
                val surfaceHeight = surfaceView.height.toFloat()
                val bitmapWidth = finalBitmap?.width ?: 0
                val bitmapHeight = finalBitmap?.height ?: 0

                val scaleX = surfaceWidth / bitmapWidth
                val scaleY = surfaceHeight / bitmapHeight
                val scale = scaleX.coerceAtMost(scaleY)

                // Calculate the scaled dimensions of the bitmap
                val scaledWidth = (bitmapWidth * scale).toInt()
                val scaledHeight = (bitmapHeight * scale).toInt()

                // Calculate the destination rectangle for the scaled bitmap
                val destLeft = (surfaceWidth - scaledWidth) / 2
                val destTop = (surfaceHeight - scaledHeight) / 2
                val destRight = destLeft + scaledWidth
                val destBottom = destTop + scaledHeight
                val destRect = Rect(destLeft.toInt(), destTop.toInt(), destRight.toInt(), destBottom.toInt())

                // Draw the scaled bitmap on the canvas
                finalBitmap?.let {
                    canvas.drawBitmap(it, null, destRect, Paint())
                }
            }
        } finally {
            surfaceHolder.unlockCanvasAndPost(canvas)
        }
    }


    private fun getOutputFilePath(): String {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())

        // Create a file name with the current timestamp
        val fileName = "video_$timeStamp.mp4"

        // Replace with your desired directory path
        val directory = Environment.getExternalStorageDirectory().absolutePath + "/recordings/"

        // Create the directory if it doesn't exist
        val dir = File(directory)
        if (!dir.exists()) {
            dir.mkdirs()
        }

        // Return the complete file path
        return directory + fileName
    }
    // Start the recording
    private fun startRecording() {

        // Create a new MediaRecorder instance
        mediaRecorder = MediaRecorder()

        // Configure the audio source, output format, and encoding parameters
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC)
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE)
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.DEFAULT)

        val outputFilePath = getOutputFilePath()
        mediaRecorder.setOutputFile(outputFilePath)

        try {
            // Configure the MediaRecorder

            mediaRecorder.prepare()
//            mediaRecorder.setPreviewDisplay(surfaceHolder.surface)
            // Start the recording
            mediaRecorder.start()
            isRecording = true


            viewBinding.videoCaptureButton.text = "Stop Recording"
        } catch (e: IOException) {
            // Handle any errors
            e.printStackTrace()
            Toast.makeText(this, "Failed to start recording", Toast.LENGTH_SHORT).show()
        }
    }

    // Stop the recording
    private fun stopRecording() {
        try {
            // Stop the recording
            mediaRecorder.stop()

            // Reset the MediaRecorder
            mediaRecorder.reset()

            mediaRecorder.release()
            isRecording = false
            viewBinding.videoCaptureButton.text = "Start Recording"
            Toast.makeText(this, "Recording saved", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            // Handle any errors
            e.printStackTrace()
            Toast.makeText(this, "Failed to stop recording", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            imageCapture = ImageCapture.Builder().build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

//            imageAnalyzer.sett

            imageAnalyzer.setAnalyzer(
                ContextCompat.getMainExecutor(this)
            ) { image: ImageProxy -> analyzer(image) }




            // Select back camera as a default
            @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
            val cameraSelector = CameraSelector.Builder()
                .addCameraFilter {
                    it.filter { camInfo ->
                        val level = Camera2CameraInfo.from(camInfo)
                            .getCameraCharacteristic(
                                CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL
                            )
                        level == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3
                    }
                }.requireLensFacing(LENS_FACING_BACK).build()


            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector,  imageAnalyzer
                )

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }


    private fun nonMaxSuppression(
        boundingBoxes: MutableList<FloatArray>,
        overlapThresh: Float
    ): MutableList<FloatArray> {

        val selectedBoxes = mutableListOf<FloatArray>()

        // Loop over all bounding boxes
        while (boundingBoxes.isNotEmpty()) {
            // Select the bounding box with the highest confidence score
            val maxBox = boundingBoxes.maxByOrNull { it[4] } ?: break
            selectedBoxes.add(maxBox)

            // Remove the selected box from the list
            boundingBoxes.remove(maxBox)

            // Calculate overlap with other bounding boxes
            val overlaps = mutableListOf<FloatArray>()
            for (box in boundingBoxes) {
                val overlap = calculateOverlap(maxBox, box)
                if (overlap > overlapThresh) {
                    overlaps.add(box)
                }
            }

            // Remove all bounding boxes that overlap with the selected box
            boundingBoxes.removeAll(overlaps)
        }

        return selectedBoxes
    }


    private fun calculateOverlap(boxA: FloatArray, boxB: FloatArray): Float {
        val xA = maxOf(boxA[0], boxB[0])
        val yA = maxOf(boxA[1], boxB[1])
        val xB = minOf(boxA[2], boxB[2])
        val yB = minOf(boxA[3], boxB[3])

        val interArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)

        val boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        val boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / (boxAArea + boxBArea - interArea)
    }

    private fun convertStringToList(input: String): MutableList<IntArray> {
        val pattern = Regex("""\((\d+(?:, \d+)*)\)""")
        val matches = pattern.findAll(input).toList()
        return matches.map { match ->
            val tupleString = match.groupValues[1]
            val tuple = tupleString.split(", ").map { it.toInt() }.toIntArray()
            tuple
        }.toMutableList()
    }
    private fun analyzer(imageProxy: ImageProxy) {


        val rotation = imageProxy.imageInfo.rotationDegrees
        val resizedBitmap = imageProxyToBitmap(imageProxy,rotation)
        val scaledBitmap = Bitmap.createScaledBitmap(
            resizedBitmap, inputShape[2], inputShape[2], false
        )

        val inputBuffer = createInputBuffer(scaledBitmap)

//        Log.d("BITMAP INFO", "Width : ${resizedBitmap.width} Height : ${resizedBitmap.height}")
//        Log.d("SCREEN INFO", "Width : ${viewBinding.viewImage.width} Height : ${viewBinding.viewImage.height}")


        executor.execute {
            inferenceAndPostProcess(inputBuffer, resizedBitmap.width, resizedBitmap.height)
        }

        val startTime = SystemClock.uptimeMillis()
        drawRectangleAndShow(resizedBitmap)
//        drawBitmapOnSurface()
        for (box in finalBoundingBoxes) {
            Log.d("BOX", box.contentToString())
        }
//        viewBinding.viewImage.setImageBitmap(resizedBitmap)
        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
        Log.d("TIME SPENT", "$timeSpent ms")
        imageProxy.close()
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

            // Create a destination rectangle for the blurred region
            val rect = Rect(left, top, right, bottom)

            // Create a bitmap for the blurred region
            val blurredRegion =
                Bitmap.createBitmap(rect.width(), rect.height(), Bitmap.Config.ARGB_8888)

            // Create a Canvas object for the blurred region bitmap
            val blurredCanvas = Canvas(blurredRegion)

            // Draw the portion of the original bitmap within the rectangle on the blurred canvas
            blurredCanvas.drawBitmap(bitmap, -left.toFloat(), -top.toFloat(), null)

            // Apply the blur effect to the blurred region
            val rs = RenderScript.create(this)
            val blurInput = Allocation.createFromBitmap(rs, blurredRegion)
            val blurOutput = Allocation.createTyped(rs, blurInput.type)
            val blurScript = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs))
            blurScript.setRadius(20f)
            blurScript.setInput(blurInput)
            blurScript.forEach(blurOutput)
            blurOutput.copyTo(blurredRegion)
            rs.destroy()

            // Draw the blurred region onto the canvas
            canvas.drawBitmap(blurredRegion, left.toFloat(), top.toFloat(), blurPaint)
        }
            finalBitmap = mutableBitmap

            viewBinding.viewImage.setImageBitmap(finalBitmap)
    }

    private fun createInputBuffer(resizedBitmap: Bitmap): ByteBuffer {
//        Log.d("TYPE", inputType)
        val inputBuffer = ByteBuffer.allocateDirect(inputSize * inputType.byteSize() ).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        val pixels = IntArray(inputShape[1] * inputShape[2])
        resizedBitmap.getPixels(pixels, 0, inputShape[2], 0, 0, inputShape[2], inputShape[1])

        for (pixel in pixels) {
            inputBuffer.put((pixel and 0xFF).toByte())
            inputBuffer.put((pixel shr 8 and 0xFF).toByte())
            inputBuffer.put((pixel shr 16 and 0xFF).toByte())
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy, rotation: Int): Bitmap {

        val yBuffer = imageProxy.planes[0].buffer // Y
        val uBuffer = imageProxy.planes[1].buffer // U
        val vBuffer = imageProxy.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val out = ByteArrayOutputStream(ySize + uSize + vSize)
        out.use {
            YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null).compressToJpeg(
                Rect(0, 0, imageProxy.width, imageProxy.height),
                100,
                it
            )
            val jpegData = it.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)

            if (rotation != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotation.toFloat())
                return Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )
            }

        return bitmap
        }

    }

    private fun inferenceAndPostProcess(inputBuffer: ByteBuffer, width: Int, height: Int) {
        val startTime = SystemClock.uptimeMillis()
        val outputBuffer = Array(1) {
            Array(outputShape[1]) {
                ByteArray(6)
            }
        }
        interpreter.run(inputBuffer, outputBuffer)
        val boundingBoxes = mutableListOf<FloatArray>()

        for (i in 0 until outputShape[1]) {
            val quantizedValue = outputBuffer[0][i][4]
            val objScore = (dequantizeFactor * (outputBuffer[0][i][4] - dequantizeBias)).toFloat()
            val classProb = (dequantizeFactor * (outputBuffer[0][i][5] - dequantizeBias)).toFloat()
            val confScore = classProb * objScore
//            val classProb = outputBuffer[0][i][4]
            if (objScore > 0.20f) {
//                Log.d("Class PROB", classProb.toString())
                val xPos =
                    (dequantizeFactor * (outputBuffer[0][i][0] - dequantizeBias)) * width
//                val xPos = outputBuffer[0][i][0] * width
//                Log.d("X Center", xPos.toString())
                val yPos =
                    (dequantizeFactor * (outputBuffer[0][i][1] - dequantizeBias)) * height
//                val yPos = outputBuffer[0][i][1] * height
//                Log.d("Y Center", yPos.toString())
                val widthBox =
                    (dequantizeFactor * (outputBuffer[0][i][2] - dequantizeBias)) * width
//                val widthBox = outputBuffer[0][i][2] * width
//                Log.d("WIDTH BOX", widthBox.toString())
                val heightBox =
                    (dequantizeFactor * (outputBuffer[0][i][3] - dequantizeBias)) * height
//                val heightBox = outputBuffer[0][i][3] * height
//                Log.d("HEIGHT BOX", heightBox.toString())
                val box = floatArrayOf(
                    max(0f, (xPos - widthBox / 2).toFloat()),
                    max(0f, (yPos - heightBox / 2).toFloat()),
                    min(width.toFloat(), (xPos + widthBox / 2).toFloat()),
                    min(height.toFloat(), (yPos + heightBox / 2).toFloat()),
                    objScore
                )
//                Log.d("BOX", Arrays.toString(box))
//                Log.d("JEDA", "================================================================")
                boundingBoxes.add(box)
            }
        }


        val postProcessedBoxes = nonMaxSuppression(boundingBoxes, 0.45f)
//        finalBoundingBoxes = postProcessedBoxes
        finalBoundingBoxes = updateSort(postProcessedBoxes)



        val endTime = SystemClock.uptimeMillis()
        val inferenceTime = endTime - startTime
        Log.d("Analyze", "Inference time Thread: $inferenceTime ms")
    }

    private fun updateSort(detections: List<FloatArray> = mutableListOf()): MutableList<FloatArray> {
        this.frameCount++
        val tracks = MutableList(this.objectTrackers.size) { FloatArray(5) }
        val toDelete: MutableList<Int> = mutableListOf()
//        var returnedValue: MutableList<FloatArray> = mutableListOf()
//        Log.d("ASS" ,this.objectTrackers.size.toString())

        for ((index, _) in tracks.withIndex()) {
            val position = this.objectTrackers[index].predict()
//            Log.d("Position", Arrays.toString(position))
            tracks[index] = position
            val isEmpty = position.isEmpty()
            val hasNaN = position.any { it.isNaN() }
//            val sizeNotEqual5 = position.size != 5
            if (isEmpty || hasNaN) {
                toDelete.add(index)
            }
        }
        toDelete.sortedDescending().forEach { index ->
            // Check if the index is within the range of the list
            if (index in 0 until tracks.size) {
                // Remove the element at the specified index
                tracks.removeAt(index)
                this.objectTrackers.removeAt(index)
            }
        }

        val associations = associateDetectionsToTrackers(detections, tracks, 0.3f)

        associations.first.forEach {
            this.objectTrackers[it[0]].update(detections[it[1]])
        }

        associations.second.forEach {
            val tracker = KalmanBoxTracker(detections[it])
            this.objectTrackers.add(tracker)
        }

        val returnedBBox = mutableListOf<FloatArray>()
        Log.d("TRACKER LENGTH", this.objectTrackers.size.toString())
        var i = this.objectTrackers.size
        for (tracker in this.objectTrackers.reversed()) {
            val bbox = tracker.getState()
//            if (tracker.timeSinceUpdate < 1 && (tracker.hitStreak >= this.minHits || this.frameCount <= this.minHits)) {
//                returnedBBox.add(bbox.plus((tracker.id + 1).toFloat()))
//            }
            returnedBBox.add(bbox.plus((tracker.id + 1).toFloat()))
            i -= 1
            if (tracker.timeSinceUpdate > this.maxAge) {
                this.objectTrackers.removeAt(i)
            }
        }

        Log.d("TRACKER LENGTH AFTER", this.objectTrackers.size.toString())

        return returnedBBox
    }

    private fun findMaxRowAndColSum(matrix: Array<IntArray>): Pair<Int, Int> {
        val numRows = matrix.size
        val numCols = matrix[0].size

        var maxRowSum = Int.MIN_VALUE
        var maxColSum = Int.MIN_VALUE

        for (i in 0 until numRows) {
            var rowSum = 0
            for (j in 0 until numCols) {
                rowSum += matrix[i][j]
            }
            if (rowSum > maxRowSum) {
                maxRowSum = rowSum
            }
        }

        for (j in 0 until numCols) {
            var colSum = 0
            for (i in 0 until numRows) {
                colSum += matrix[i][j]
            }
            if (colSum > maxColSum) {
                maxColSum = colSum
            }
        }

        return Pair(maxRowSum, maxColSum)
    }

    private fun findOnes(matrix: Array<IntArray>): MutableList<IntArray> {
        val locations = mutableListOf<IntArray>()

        for (i in matrix.indices) {
            for (j in matrix[i].indices) {
                if (matrix[i][j] == 1) {
                    locations.add(intArrayOf(i, j))
                }
            }
        }

        return locations
    }

    private fun associateDetectionsToTrackers(
        detections: List<FloatArray>,
        trackers: List<FloatArray>,
        iouThreshold: Float = 0.3f
    ): Triple<MutableList<IntArray>, MutableList<Int>, MutableList<Int>> {
        if (trackers.isEmpty()) {
            val unmatchedDetections = MutableList(detections.size) { it }
//            Log.d("unmatchedDetections", unmatchedDetections.toString())
            return Triple(mutableListOf(), unmatchedDetections, mutableListOf())
        }
        for (tracker in trackers){
            Log.d("TRACKER", tracker.contentToString())
        }
        for (detection in detections){
            Log.d("DETECTIONS", detection.contentToString())
        }

        val iouMatrix = computeIoUMatrix(detections, trackers)
//        Log.d()
//        iouMatrix.contentToString()
        for (iou in iouMatrix) {
            Log.d("IouMAT", iou.contentToString())
        }

        val allEmpty = iouMatrix.any { it.isEmpty() }

        var matchedIndices = mutableListOf<IntArray>()
//        iouMatrix.transpose()
//        Log.d("IoUMATRIX", iouMatrix.contentToString())
//        Log.d("MATRIX EMPTY", iouMatrix.isNotEmpty().toString())
//        Log.d("ALL EMPTY",allEmpty.toString())
        if (iouMatrix.isNotEmpty() && !allEmpty) {
            val numRows = iouMatrix.size
            val numCols = iouMatrix[0].size
            val thresholdMatrix = Array(numRows) { IntArray(numCols) }

            for (i in 0 until numRows) {
                for (j in 0 until numCols) {
//                    Log.d("IouTH", iouThreshold.toString())
//                    Log.d("IoU", "[${i},${j}] : ${iouMatrix[i][j]}")
                    if (iouMatrix[i][j] > iouThreshold) {
                        thresholdMatrix[i][j] = 1
                    } else {
                        thresholdMatrix[i][j] = 0
                    }
                }
            }
//            Log.d("TH Matrix", thresholdMatrix.toString())
            for (tM in thresholdMatrix) {
                Log.d("T", tM.contentToString())
            }



            val a = findMaxRowAndColSum(thresholdMatrix)
//            Log.d("MAXROWANDCOLUMNSUM", a.toString())

            if (a.first == 1 && a.second == 1) {
                Log.d("SINI1", "------1------")
                matchedIndices = findOnes(thresholdMatrix)
            } else {
                Log.d("SINI2", "------2------")
//                matchedIndices = hungarianAlgorithm(iouMatrix)
                val result = module.callAttr("lsa", iouMatrix).toString()
                matchedIndices = convertStringToList(result)
            }
        }

        val unmatchedDetections = mutableListOf<Int>()

        detections.forEachIndexed { i, _ ->
            val notExist = matchedIndices.none { it[1] == i }
            if (notExist) {
                unmatchedDetections.add(i)
            }
        }

        val unmatchedTrackers = mutableListOf<Int>()

        trackers.forEachIndexed { i, _ ->
            val notExist = matchedIndices.none { it[0] == i }
            if (notExist) {
                unmatchedTrackers.add(i)
            }
        }

//        Log.d("MATCHES INDICES", matchedIndices)
        for (match in matchedIndices) {
            Log.d("MATCH", match.contentToString())
        }


        for (unmatchDet in unmatchedDetections) {
            Log.d("UNMATCH DET", unmatchDet.toString())
        }
        for (unmatchTrack in unmatchedTrackers) {
            Log.d("UNMATCH TRACKQ", unmatchTrack.toString())
        }

        val matches = mutableListOf<IntArray>()

        for (it in matchedIndices) {
//            if (it[1] < 0 || it[0] < 0) continue
            if (iouMatrix[it[0]][it[1]] < iouThreshold) {
                unmatchedDetections.add(it[1])
                unmatchedTrackers.add(it[0])
            } else {
                matches.add(intArrayOf(it[0], it[1]))
            }
        }

        return Triple(matches, unmatchedDetections, unmatchedTrackers)

    }

    private fun computeIoUMatrix(detections: List<FloatArray>, trackers: List<FloatArray>): Array<FloatArray> {
        val numDetections = detections.size
        val numTrackers = trackers.size

        // Initialize the IoU matrix
        val iouMatrix = Array(numTrackers) { FloatArray(numDetections) }

        for (i in 0 until numTrackers) {
            val tracker = trackers[i]
            val trackerArea = (tracker[2] - tracker[0]) * (tracker[3] - tracker[1])

            for (j in 0 until numDetections) {
                val detection = detections[j]
                val detectionArea = (detection[2] - detection[0]) * (detection[3] - detection[1])

                // Compute the intersection rectangle
                val xA = max(tracker[0], detection[0])
                val yA = max(tracker[1], detection[1])
                val xB = min(tracker[2], detection[2])
                val yB = min(tracker[3], detection[3])
                val interArea = max(0.0F, xB - xA) * max(0.0F, yB - yA)

                // Compute the union area
                val unionArea = trackerArea + detectionArea - interArea

                // Compute the IoU and store it in the matrix
                val iou = interArea / unionArea
//                Log.d("InterArea", interArea.toString())
//                Log.d("UnionArea", unionArea.toString())

//                for (row in iouMatrix) {
//                    Log.d("COST", row.toString())
//                }
                iouMatrix[i][j] = iou
            }
        }

        return iouMatrix
    }




    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("best-200e-yolov5n-416-int8.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }



    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
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


    @SuppressLint("MissingSuperCall")
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
}
