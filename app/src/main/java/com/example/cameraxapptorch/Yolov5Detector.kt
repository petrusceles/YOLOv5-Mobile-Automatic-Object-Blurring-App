package com.example.cameraxapptorch

import android.graphics.Bitmap
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

class Yolov5Detector (tfliteModel: MappedByteBuffer, options: Interpreter.Options) {

    private var interpreter: Interpreter = Interpreter(tfliteModel,options)


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
    private var outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(outputShape[1]*6).apply {
        order(ByteOrder.nativeOrder())
        rewind()
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
    }

    fun inferenceAndPostProcess(width: Int, height: Int, bitmap: Bitmap): MutableList<FloatArray> {
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

        for (i in 0 until outputShape[1]) {
            val objScore = out[i][4]
            if (objScore >= Yolov5Model.getConfThreshold()) {
                if (out[i][2] > 0.9f && out[i][3] > 0.9f) {
                    continue
                }
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
        return nonMaxSuppression(boundingBoxes)
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