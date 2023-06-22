package com.example.cameraxapptorch

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
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
import com.example.cameraxapptorch.databinding.ActivityMainBinding
import org.apache.commons.math3.linear.*
import org.apache.commons.math3.optim.linear.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
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
import kotlin.math.roundToInt


private val IMAGE_MEAN = 0f
private val IMAGE_STD = 255.0f
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
    @SuppressLint("SimpleDateFormat")
    private val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss")

    private var objectTrackers: MutableList<KalmanBoxTracker> = mutableListOf()
    private val maxAge: Int = 5
    private var frameCount: Int = 0
    private lateinit var module: PyObject

    private var finalBoundingBoxes: MutableList<FloatArray> = mutableListOf()
    private var untrackedBoundingBoxes: MutableList<FloatArray> = mutableListOf()

    private var finalBitmap: Bitmap? = null

    private var isCapture = false
    private val REQUEST_CODE = 22

    private var inputScale: Float = 0.0f
    private var inputZeroPoint: Int = 0
    private var outputScale: Float = 0.0f
    private var outputZeroPoint: Int = 0

    private var counter: Int = 0

    private var outputBox: Int = 0

    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: ByteBuffer
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

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), REQUEST_CODE)
        }
        val windowInsetsController =
            WindowCompat.getInsetsController(window, window.decorView)
        windowInsetsController.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE

        windowInsetsController.hide(WindowInsetsCompat.Type.systemBars())
        val tfliteModel: MappedByteBuffer = Yolov5Model.getMappedByteBuffer()
        val options = Interpreter.Options()
        options.useNNAPI = true
        options.numThreads = 2
        interpreter = Interpreter(tfliteModel, options)

        inputTensor = interpreter.getInputTensor(0)

        inputScale = inputTensor.quantizationParams().scale;
        inputZeroPoint = inputTensor.quantizationParams().zeroPoint;

        inputShape = inputTensor.shape()  // [batch_size, height, width, channels]
        inputType = inputTensor.dataType()

        inputSize = inputShape[1] * inputShape[2] * inputShape[3]  // height * width * channels


        outputTensor = interpreter.getOutputTensor(0)
        outputScale = outputTensor.quantizationParams().scale;
        outputZeroPoint = outputTensor.quantizationParams().zeroPoint;

        outputShape = outputTensor.shape()  // [batch_size, height, width, channels]
        outputType = outputTensor.dataType()

        outputSize = outputShape[0] * outputShape[1] * outputShape[2]  // height * width * channels
        interpreter = Interpreter(tfliteModel, options)
        outputBox = outputShape[1]

        inputBuffer = ByteBuffer.allocateDirect(inputSize * inputType.byteSize() ).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        outputBuffer = ByteBuffer.allocateDirect(outputBox*6).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        val py = Python.getInstance()
        this.module = py.getModule("lsa")

        viewBinding.captureButton.setOnClickListener {
            isCapture = !isCapture
            if (isCapture) {
                viewBinding.captureButton.text = "Stop Capture"
            } else {
                viewBinding.captureButton.text = "Start Capture"
            }
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
        boundingBoxes: MutableList<FloatArray>
    ): MutableList<FloatArray> {

        val selectedBoxes = mutableListOf<FloatArray>()
        while (boundingBoxes.isNotEmpty()) {
            val maxBox = boundingBoxes.maxByOrNull { it[4] } ?: break
            selectedBoxes.add(maxBox)

            boundingBoxes.remove(maxBox)

            val overlaps = mutableListOf<FloatArray>()
            for (box in boundingBoxes) {
                val overlap = calculateOverlap(maxBox, box)
                if (overlap > Yolov5Model.getIouThreshold()) {
                    overlaps.add(box)
                }
            }

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
        createInputBuffer(scaledBitmap)
        counter += 1

        executor.execute {
            inferenceAndPostProcess(resizedBitmap.width, resizedBitmap.height)
        }

        val startTime = SystemClock.uptimeMillis()
        drawRectangleAndShow(resizedBitmap)
        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
        Log.d("TIME SPENT", "$timeSpent ms")
        imageProxy.close()
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
//        Log.d("GRANTED", isGranted.toString())
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
        if (isCapture) {
            val currentTime = dateFormat.format(Date()).replace(":", ".")
            val folder = File(getExternalFilesDir(null), "${Yolov5Model.getFolderMain()}-${Yolov5Model.getFolderPrefix()}")
            folder.mkdirs()
            val outputTxtFile = File(folder,"$currentTime.txt")
            val outputPngFile = File(folder,"$currentTime.jpg")
            saveBoundingBoxes(outputTxtFile, finalBoundingBoxes, bitmap.width, bitmap.height) // Save bounding box data
            saveImage(outputPngFile, bitmap) // Save image data

            if (Yolov5Model.getIsTracking() && Yolov5Model.getIsSaveUntracked()) {
                val outputTxtFileUntracked = File(folder, "$currentTime-untracked.txt")
                saveBoundingBoxes(outputTxtFileUntracked, untrackedBoundingBoxes, bitmap.width, bitmap.height)
            }
        }

    }

    private fun createInputBuffer(resizedBitmap: Bitmap) {
//        Log.d("TYPE", inputType)
        inputBuffer.rewind()
        val pixels = IntArray(inputShape[1] * inputShape[2])
        resizedBitmap.getPixels(pixels, 0, inputShape[2], 0, 0, inputShape[2], inputShape[1])
        inputBuffer.rewind()
        for (pixel in pixels) {
            inputBuffer.put(
                (((pixel shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inputScale + inputZeroPoint).toInt()
                    .toByte()
            )
            inputBuffer.put(
                (((pixel shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inputScale + inputZeroPoint).toInt()
                    .toByte()
            )
            inputBuffer.put(
                (((pixel and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inputScale + inputZeroPoint).toInt()
                    .toByte()
            )
        }
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

    private fun inferenceAndPostProcess(width: Int, height: Int) {
        var outputMap: MutableMap<Int, Any> = HashMap()
        val boundingBoxes = mutableListOf<FloatArray>()
        outputBuffer.rewind()
        outputMap.put(0, outputBuffer)
        val inputArray = arrayOf<Any>(inputBuffer)
        val startTime = SystemClock.uptimeMillis()
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
        val endTime = SystemClock.uptimeMillis()
        val inferenceTime = endTime - startTime
        Log.d("Analyze", "Inference time Thread: $inferenceTime ms")
        val byteBuffer = outputMap[0] as ByteBuffer?
        byteBuffer!!.rewind()

        val out = Array(outputBox) {
            FloatArray(6)
        }

        for (i in 0 until outputBox) {
            for (j in 0 until 6) {
                out[i][j] = outputScale * ((byteBuffer.get().toInt() and 0xFF) - outputZeroPoint)
            }
        }

        for (i in 0 until outputBox) {
            val objScore = out[i][4]
            if (objScore >= Yolov5Model.getConfThreshold()) {
                val xPos = out[i][0] * width
                val yPos = out[i][1] * height
                val widthBox = out[i][2] * width * 1.5f
                val heightBox = out[i][3] * height * 1.5f
                val box = floatArrayOf(
                    max(0f, (xPos - widthBox / 2)),
                    max(0f, (yPos - heightBox / 2)),
                    min(width.toFloat(), (xPos + widthBox / 2)),
                    min(height.toFloat(), (yPos + heightBox / 2)),
                    objScore
                )
                boundingBoxes.add(box)
            }
        }

        for (box in boundingBoxes) {
            Log.d("BOUNDING BOX", box.contentToString())
        }


        untrackedBoundingBoxes = nonMaxSuppression(boundingBoxes)


        if (Yolov5Model.getIsTracking()) {
            Log.d("TRACKING", "TRUE")
            finalBoundingBoxes = updateSort(untrackedBoundingBoxes)
        } else {
            Log.d("TRACKING", "FALSE")
            finalBoundingBoxes = untrackedBoundingBoxes
        }

        counter = 0
    }

    private fun updateSort(detections: List<FloatArray> = mutableListOf()): MutableList<FloatArray> {
        this.frameCount++
        val tracks = MutableList(this.objectTrackers.size) { FloatArray(5) }
        val toDelete: MutableList<Int> = mutableListOf()

        for ((index, _) in tracks.withIndex()) {
            val position = this.objectTrackers[index].predict()
            tracks[index] = position
            val isEmpty = position.isEmpty()
            val hasNaN = position.any { it.isNaN() }
            if (isEmpty || hasNaN) {
                toDelete.add(index)
            }
        }
        toDelete.sortedDescending().forEach { index ->
            if (index in 0 until tracks.size) {
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

        var i = this.objectTrackers.size
        for (tracker in this.objectTrackers.reversed()) {
            val bbox = tracker.getState()

            returnedBBox.add(bbox.plus((tracker.id + 1).toFloat()))
            i -= 1
            if (tracker.timeSinceUpdate > this.maxAge) {
                this.objectTrackers.removeAt(i)
            }
        }


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
//        for (tracker in trackers){
//            Log.d("TRACKER", tracker.contentToString())
//        }
//        for (detection in detections){
//            Log.d("DETECTIONS", detection.contentToString())
//        }

        val iouMatrix = computeIoUMatrix(detections, trackers)
//        Log.d()
//        iouMatrix.contentToString()
//        for (iou in iouMatrix) {
//            Log.d("IouMAT", iou.contentToString())
//        }

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
//                Log.d("SINI1", "------1------")
                matchedIndices = findOnes(thresholdMatrix)
            } else {
//                Log.d("SINI2", "------2------")
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


//        for (unmatchDet in unmatchedDetections) {
//            Log.d("UNMATCH DET", unmatchDet.toString())
//        }
//        for (unmatchTrack in unmatchedTrackers) {
//            Log.d("UNMATCH TRACKQ", unmatchTrack.toString())
//        }

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
