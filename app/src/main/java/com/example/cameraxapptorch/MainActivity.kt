package com.example.cameraxapptorch

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
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


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

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
    private val maxAge: Int=1
    private val minHits: Int=3
    private val iouThreshold: Float = 0.3f
    private var frameCount:Int = 0
//    self.max_age = max_age
//    self.min_hits = min_hits
//    self.iou_threshold = iou_threshold
//    self.trackers = []
//    self.frame_count = 0

//    private var TrackedObjects = listOf<Tracked>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

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
        viewBinding.videoCaptureButton.setOnClickListener { captureVideo() }

        cameraExecutor = Executors.newSingleThreadExecutor()


    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun
                        onImageSaved(output: ImageCapture.OutputFileResults){
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            }
        )
    }

    // Implements VideoCapture use case, including start and stop capturing.
    private fun captureVideo() {
        val videoCapture = this.videoCapture ?: return

        viewBinding.videoCaptureButton.isEnabled = false

        val curRecording = recording
        if (curRecording != null) {
            // Stop the current recording session.
            curRecording.stop()
            recording = null
            return
        }

        // create and start a new recording session
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/CameraX-Video")
            }
        }

        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()
        recording = videoCapture.output
            .prepareRecording(this, mediaStoreOutputOptions)
            .apply {
                if (PermissionChecker.checkSelfPermission(this@MainActivity,
                        Manifest.permission.RECORD_AUDIO) ==
                    PermissionChecker.PERMISSION_GRANTED)
                {
                    withAudioEnabled()
                }
            }
            .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                when(recordEvent) {
                    is VideoRecordEvent.Start -> {
                        viewBinding.videoCaptureButton.apply {
                            text = getString(R.string.stop_capture)
                            isEnabled = true
                        }
                    }
                    is VideoRecordEvent.Finalize -> {
                        if (!recordEvent.hasError()) {
                            val msg = "Video capture succeeded: " +
                                    "${recordEvent.outputResults.outputUri}"
                            Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT)
                                .show()
                            Log.d(TAG, msg)
                        } else {
                            recording?.close()
                            recording = null
                            Log.e(TAG, "Video capture ends with error: " +
                                    "${recordEvent.error}")
                        }
                        viewBinding.videoCaptureButton.apply {
                            text = getString(R.string.start_capture)
                            isEnabled = true
                        }
                    }
                }
            }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()



            // Preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

//            val camera2Interop = Camera2Interop.Extender
//            camera2Interop.setTargetFps(30)


            imageCapture = ImageCapture.Builder().build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
//                .setTargetRotation(ROTATION_90)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

//            imageAnalyzer.sett

            imageAnalyzer.setAnalyzer(ContextCompat.getMainExecutor(this)
            ) { image: ImageProxy -> analyzer(image) }

            val selector = QualitySelector
                .from(
                    Quality.UHD,
                    FallbackStrategy.higherQualityOrLowerThan(Quality.SD)
                )

            val recorder = Recorder.Builder()
                .setQualitySelector(selector)
                .build()



            videoCapture = VideoCapture.withOutput(recorder)

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
                    this, cameraSelector,preview, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }


    private fun nonMaxSuppression(boundingBoxes: MutableList<FloatArray>, overlapThresh: Float): MutableList<FloatArray> {

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

    private fun analyzer (imageProxy: ImageProxy) {

        viewBinding.overlay.clearRect()

        val startTime = SystemClock.uptimeMillis()

        val resizedBitmap = imageProxyToBitmap(imageProxy)

        val inputBuffer = createInputBuffer(resizedBitmap)

        executor.execute{
            inferenceAndPostProcess(inputBuffer)
        }

        val endTime = SystemClock.uptimeMillis()
        val inferenceTime = endTime - startTime
        Log.d("Analyze", "Inference time: $inferenceTime ms")

        imageProxy.close()
    }

    private fun createInputBuffer(resizedBitmap: Bitmap) : ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(inputSize * inputType.byteSize()).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }
        for (y in 0 until inputShape[1]) {
            for (x in 0 until inputShape[2]) {
                val pixel = resizedBitmap.getPixel(x, y)
                inputBuffer.put((pixel shr 16 and 0xFF).toByte())
                inputBuffer.put((pixel shr 8 and 0xFF).toByte())
                inputBuffer.put((pixel and 0xFF).toByte())
            }
        }
        inputBuffer.rewind()
        return inputBuffer
    }
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
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
        val out = ByteArrayOutputStream()
        YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null).compressToJpeg(
            Rect(0, 0, imageProxy.width, imageProxy.height),
            100,
            out
        )
        val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())

        val matrix = Matrix()
        matrix.postRotate(90f)
        val rotatedBitmap =
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        return Bitmap.createScaledBitmap(rotatedBitmap, inputShape[2], inputShape[2], false)
    }

    private fun inferenceAndPostProcess(inputBuffer: ByteBuffer) {
        val outputBuffer = Array(1) {
            Array(outputShape[1]) {
                ByteArray(6)
            }
        }
        interpreter.run(inputBuffer,outputBuffer)
        val boundingBoxes = mutableListOf<FloatArray>()
        for (i in 0 until outputShape[1]) {
            val classProb = (0.011072992347180843 * (outputBuffer[0][i][4] - 2))
            if (classProb >= 0.6) {
                val xPos = (0.011072992347180843 * (outputBuffer[0][i][0]  - 2)) * viewBinding.viewFinder.width
                val yPos = (0.011072992347180843 * (outputBuffer[0][i][1]  - 2)) * viewBinding.viewFinder.height
                val width = (0.011072992347180843 * (outputBuffer[0][i][2]  - 2)) * viewBinding.viewFinder.width
                val height = (0.011072992347180843 * (outputBuffer[0][i][3]  - 2)) * viewBinding.viewFinder.height
                boundingBoxes.add(floatArrayOf(
                    max(0f, (xPos - width / 2).toFloat()),
                    max(0f, (yPos - height / 2).toFloat()),
                    min(viewBinding.viewFinder.width.toFloat(), (xPos + width / 2).toFloat()),
                    min(viewBinding.viewFinder.height.toFloat(), (yPos + height / 2).toFloat()),
                    classProb.toFloat()
                ))
            }
        }


        val postProcessedBoxes = nonMaxSuppression(boundingBoxes, 0.1f)

        for (box in postProcessedBoxes) {
            Log.d("BOX", Arrays.toString(box))
        }

        viewBinding.overlay.setRect(postProcessedBoxes)
    }

    private fun updateSort(detections: List<FloatArray> = mutableListOf()) {
        this.frameCount++
        var tracks = MutableList(this.objectTrackers.size) { FloatArray(5) }
        var toDelete: MutableList<Int> = mutableListOf()
        var returnedValue: MutableList<FloatArray> = mutableListOf()

        for ((index, _) in tracks.withIndex()) {
            var position = this.objectTrackers[index].predict()
            tracks[index] = position
            val isEmpty = position.isEmpty()
            val hasNaN = position.any { it.isNaN() }
            val sizeNotEqual5 = position.size != 5
            if (isEmpty || hasNaN || sizeNotEqual5) {
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
                iouMatrix[i][j] = iou
            }
        }

        return iouMatrix
    }

    fun linearSumAssignment(costMatrix: Array<FloatArray>): IntArray {
        val numRows = costMatrix.size
        val numCols = costMatrix[0].size
        val assignments = IntArray(numRows)

        // Step 1: Subtract the minimum value in each row from all elements in that row
        for (i in 0 until numRows) {
            val rowMin = costMatrix[i].minOrNull() ?: 0f
            for (j in 0 until numCols) {
                costMatrix[i][j] -= rowMin
            }
        }

        // Step 2: Subtract the minimum value in each column from all elements in that column
        for (j in 0 until numCols) {
            val colMin = (0 until numRows).map { costMatrix[it][j] }.minOrNull() ?: 0f
            for (i in 0 until numRows) {
                costMatrix[i][j] -= colMin
            }
        }

        // Step 3: Cover all rows and columns that contain zeros
        val rowCovered = BooleanArray(numRows)
        val colCovered = BooleanArray(numCols)
        for (i in 0 until numRows) {
            for (j in 0 until numCols) {
                if (costMatrix[i][j] == 0f && !rowCovered[i] && !colCovered[j]) {
                    rowCovered[i] = true
                    colCovered[j] = true
                }
            }
        }

        var numCovered = rowCovered.count { it }
        while (numCovered < numRows) {
            // Step 4: Find an uncovered zero and prime it
            var i = 0
            while (rowCovered[i]) {
                i++
            }
            var j = 0
            while (costMatrix[i][j] != 0f || colCovered[j]) {
                j++
            }
            assignments[i] = j
            var primeRow = i
            var primeCol = j

            // Step 5: Find a non-covered zero in the same row and prime it
            var done = false
            while (!done) {
                var j2 = 0
                while (j2 < numCols && (primeRow != i || costMatrix[primeRow][j2] == 0f || colCovered[j2])) {
                    j2++
                }
                if (j2 == numCols) {
                    done = true
                } else {
                    primeCol = j2
                    rowCovered[primeRow] = true
                    colCovered[primeCol] = false
                    val index = assignments.indexOf(primeCol)
                    if (index >= 0) {
                        primeRow = index
                        done = false
                    } else {
                        done = true
                    }
                }
            }

            if (numCovered == numRows) {
                break
            }

            // Step 6: Augment the path starting at the primed zero and alternating between primed and starred zeros
            val path = mutableListOf<Pair<Int, Int>>()
            path.add(Pair(primeRow, primeCol))
            var done2 = false
            while (!done2) {
                val row = path.last().first
                val col = path.last().second
                if (rowCovered[row]) {
                    val col2 = assignments[row]
                    path.add(Pair(row, col2))
                } else {
                    done2 = true
                }
                if (!done2) {
                    val row2 = path.last().first
                    val col2 = path.last().second
                    if (colCovered[col2]) {
                        val row3 = starInCol(costMatrix, col2)
                        path.add(Pair(row3, col2))
                    } else {
                        done2 = true
                    }
                }
            }

            // Step 7: Augment the path by unstarred zeros along the path, and then remove the stars from starred zeros and re-cover rows and columns
            for (pair in path) {
                if (assignments[pair.first] == pair.second) {
                    colCovered[pair.second] = true
                    rowCovered[pair.first] = false
                } else {
                    colCovered[assignments[pair.first]] = false
                    rowCovered[pair.first] = true
                }
                assignments[pair.first] = pair.second
            }
            for (i in 0 until numRows) {
                for (j in 0 until numCols) {
                    if (costMatrix[i][j] == 0f && !rowCovered[i] && !colCovered[j]) {
                        rowCovered[i] = true
                        colCovered[j] = true
                    }
                }
            }

            numCovered = rowCovered.count { it }
        }

        return assignments
    }

    private fun starInCol(costMatrix: Array<FloatArray>, col: Int): Int {
        for (i in costMatrix.indices) {
            if (costMatrix[i][col] == 0f) {
                return i
            }
        }
        throw IllegalArgumentException("No star found")
    }




    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("person-416-int8.tflite")
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