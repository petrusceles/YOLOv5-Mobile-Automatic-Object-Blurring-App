package com.example.cameraxapptorch

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.Build
import android.os.SystemClock
import android.util.Log
import com.chaquo.python.PyObject
import androidx.annotation.RequiresApi
import androidx.renderscript.Allocation
import androidx.renderscript.Element
import androidx.renderscript.RenderScript
import androidx.renderscript.ScriptIntrinsicBlur
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.example.cameraxapptorch.Yolov5Model.IMAGE_MEAN
import com.example.cameraxapptorch.Yolov5Model.IMAGE_STD
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.HashMap
import kotlin.math.max
import kotlin.math.min

class Yolov5Detector (tfliteModel: MappedByteBuffer, options: Interpreter.Options, context: Context) {

    private var interpreter: Interpreter = Interpreter(tfliteModel,options)

    private var contextParent = context
    private var inputTensor: Tensor = interpreter.getInputTensor(0)
    var inputShape: IntArray = inputTensor.shape() // [batch_size, height, width, channels]
    private var inputType: DataType = inputTensor.dataType()
    private var inputSize: Int = inputShape[1] * inputShape[2] * inputShape[3]

    private var inputScale: Float = inputTensor.quantizationParams().scale;
    private var inputZeroPoint: Int = inputTensor.quantizationParams().zeroPoint;

    private var outputTensor: Tensor = interpreter.getOutputTensor(0)
    var outputShape: IntArray = outputTensor.shape() // [batch_size, height, width, channels]
    private var outputType: DataType = outputTensor.dataType()
    private var outputSize: Int = outputShape[0] * outputShape[1] * outputShape[2]

    private var outputScale: Float = outputTensor.quantizationParams().scale
    private var outputZeroPoint: Int = outputTensor.quantizationParams().zeroPoint;

    private var inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(inputSize * inputType.byteSize() ).apply {
        order(ByteOrder.nativeOrder())
        rewind()
    }

    private lateinit var inputBufferProcessing: ByteBuffer
    private var outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(outputShape[1]*6).apply {
        order(ByteOrder.nativeOrder())
        rewind()
    }

    private var finalBoundingBoxes: MutableList<FloatArray> = mutableListOf()
    private var untrackedBoundingBoxes: MutableList<FloatArray> = mutableListOf()
    private lateinit var currentBitmap: Bitmap
    private lateinit var mutableBitmap: Bitmap




    private var py:Python
    private var sortTracker:Sort

    private var startDrawing: Boolean = false

    init {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(contextParent))
        }
        py = Python.getInstance()
        sortTracker = Sort(py.getModule("lsa"))
    }

    fun getMutableBitmap():Bitmap {
        return mutableBitmap
    }

    fun getCurrentBitmap():Bitmap {
        return currentBitmap!!
    }

    fun setCurrentBitmap(_currentBitmap:Bitmap) {
        currentBitmap = _currentBitmap
    }

    fun getFinalBoundingBoxes():MutableList<FloatArray> {
        return finalBoundingBoxes
    }
    fun getUntrackedBoundingBoxes():MutableList<FloatArray> {
        return untrackedBoundingBoxes
    }

    fun isInputBufferProcessingFilled(): Boolean {
        return inputBufferProcessing.position() >= inputBufferProcessing.limit()
    }

    fun isCurrentBitmapInitialized(): Boolean {
        return ::currentBitmap.isInitialized
    }


    fun createInputBuffer(scaledBitmap: Bitmap) {
        val pixels = IntArray(inputShape[1] * inputShape[2])
        scaledBitmap.getPixels(pixels, 0, inputShape[2], 0, 0, inputShape[2], inputShape[1])
        inputBuffer.clear()
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
        inputBuffer.rewind()
//        inputBufferProcessing = inputBuffer.duplicate()
    }

    @RequiresApi(Build.VERSION_CODES.S)
    fun inferenceAndPostProcess() {
        val width = currentBitmap.width
        val height = currentBitmap.height
        val startTime = SystemClock.uptimeMillis()

        val boundingBoxes = mutableListOf<FloatArray>()

        val outputMap: MutableMap<Int, Any> = HashMap()
        outputBuffer.clear()
        outputMap[0] = outputBuffer
        val inputArray = arrayOf<Any>(inputBuffer)

        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)

        val byteBuffer = outputMap[0] as ByteBuffer?
        byteBuffer?.rewind()

        val out = Array(outputShape[1]) {
            FloatArray(6)
        }

        val byteBufferLimit = byteBuffer?.limit() ?: 0
        for (i in 0 until outputShape[1]) {
            for (j in 0 until 6) {
                val byteValue = if ((byteBuffer?.position() ?: 0) < byteBufferLimit) {
                    byteBuffer?.get() ?: 0
                } else {
                    0
                }
                out[i][j] = outputScale * ((byteValue.toInt() and 0xFF) - outputZeroPoint)
            }
        }

        val resizedWidth = inputShape[1]
        val resizedHeight = inputShape[1]


        val originalHeightInResized = min(height.toFloat() * (resizedWidth.toFloat()/width.toFloat()), resizedHeight.toFloat())


        val originalWidthInResized = min(width.toFloat() * (resizedHeight.toFloat()/height.toFloat()), resizedWidth.toFloat())

        Log.d("RESIZED SIZE", "W ${originalWidthInResized}")

        for (i in 0 until outputShape[1]) {
            val objScore = out[i][4]
            if (objScore >= Yolov5Model.getConfThreshold()) {
                if (out[i][2] > 0.9f && out[i][3] > 0.9f) {
                    continue
                }
                val xPos = ((2*resizedWidth*out[i][0] - resizedWidth + originalWidthInResized)*width)/(2*originalWidthInResized)
                val yPos = ((2*resizedHeight*out[i][1] - resizedHeight + originalHeightInResized)*height)/(2*originalHeightInResized)

                val widthBox = ((out[i][2] * resizedWidth)/originalWidthInResized)*width*1.4f
                val heightBox = ((out[i][3] * resizedHeight)/originalHeightInResized)*height*1.4f

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
            Log.d("BOXES", box.contentToString())
        }


        untrackedBoundingBoxes = nonMaxSuppression(boundingBoxes)

        finalBoundingBoxes = if (Yolov5Model.getIsTracking()) {
            sortTracker.updateSort(untrackedBoundingBoxes)
        } else {
            untrackedBoundingBoxes
        }
        currentBitmap.let { drawRectangleAndShow(it) }

//        inputBufferProcessing.clear()
        val timeSpent = (SystemClock.uptimeMillis() - startTime).toInt()
        Log.d("INFERENCE TIME", "$timeSpent ms")
    }

    @RequiresApi(Build.VERSION_CODES.S)
    private fun drawRectangleAndShow(bitmap: Bitmap) {
        var mutableBitmapTemp = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmapTemp)
        val blurPaint = Paint().apply {
            maskFilter = BlurMaskFilter(7f, BlurMaskFilter.Blur.NORMAL)
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

            val blurredRegion =
                Bitmap.createBitmap(rect.width(), rect.height(), Bitmap.Config.ARGB_8888)
            val blurredCanvas = Canvas(blurredRegion)
            blurredCanvas.drawBitmap(bitmap, -left.toFloat(), -top.toFloat(), null)
            val rs = RenderScript.create(contextParent)
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
        mutableBitmap = mutableBitmapTemp
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

}