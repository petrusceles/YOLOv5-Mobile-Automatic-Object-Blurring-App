package com.example.cameraxapptorch

import android.util.Log
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector
import org.apache.commons.math3.filter.MeasurementModel
import org.apache.commons.math3.filter.ProcessModel
import org.apache.commons.math3.filter.DefaultMeasurementModel
import org.apache.commons.math3.filter.DefaultProcessModel
import org.apache.commons.math3.filter.KalmanFilter
import kotlin.math.sqrt

class KalmanBoxTracker(
    boundingBox: FloatArray?
) {

    private val F: RealMatrix = Array2DRowRealMatrix(arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0, 0.01, 0.0),
        doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01),
        doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ))

    private val H: RealMatrix = Array2DRowRealMatrix(arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    ))

    private val P: RealMatrix = Array2DRowRealMatrix(7, 7).apply {
        setEntry(0, 0, 10.0)
        setEntry(1, 1, 10.0)
        setEntry(2, 2, 10.0)
        setEntry(3, 3, 10.0)
        setEntry(4, 4, 10000.0)
        setEntry(5, 5, 10000.0)
        setEntry(6, 6, 10000.0)
//        scalarMultiply(10.0)
    }

    private var x: RealVector

    private val Q: RealMatrix = Array2DRowRealMatrix(7, 7).apply {
        setEntry(0, 0, 1.0)
        setEntry(1, 1, 1.0)
        setEntry(2, 2, 1.0)
        setEntry(3, 3, 1.0)
        setEntry(4, 4, 0.01)
        setEntry(5, 5, 0.01)
        setEntry(6, 6, 0.0001)
//        setSubMatrix(Array2DRowRealMatrix(Array(7) { i -> doubleArrayOf(0.0) }).scalarMultiply(0.01).data, 0, 0)
    }

    private val R: RealMatrix = Array2DRowRealMatrix(4, 4).apply {
        setEntry(0, 0, 1.0)
        setEntry(1, 1, 1.0)
        setEntry(2, 2, 10.0)
        setEntry(3, 3, 10.0)
    }

    private fun getBoxFeatures(box: FloatArray): RealVector {
        val x1 = box[0]
        val y1 = box[1]
        val x2 = box[2]
        val y2 = box[3]
//        Log.d("BOX BEFORE", box.contentToString())
        val width = x2 - x1
        val height = y2 - y1
        val xCenter = x1 + 0.5 * width
        val yCenter = y1 + 0.5 * height
        val scale = width * height
        val aspectRatio = width / height
        return ArrayRealVector(doubleArrayOf(xCenter, yCenter, scale.toDouble(),
            aspectRatio.toDouble()
        ))
    }

    private fun convertBBox(bBox: DoubleArray): FloatArray {
        val width = sqrt(bBox[2] * bBox[3])
        val height = bBox[2]/width
        val xCenter = bBox[0]
        val yCenter = bBox[1]
        val x1 = xCenter - (width / 2)
        val y1 = yCenter - (height / 2)
        val x2 = xCenter + (width / 2)
        val y2 = yCenter + (height / 2)
        return floatArrayOf(x1.toFloat(), y1.toFloat(), x2.toFloat(), y2.toFloat())
    }

    private var processModel: DefaultProcessModel

    private var measurementModel: MeasurementModel

    var timeSinceUpdate = 0.0f

    private var history = mutableListOf<FloatArray>()

    private var hits :Int = 0

    var hitStreak: Int = 0

    private var age :Int = 0

    companion object {
        private var count = 0
    }
    var id: Int = 0

    private var kalmanFilter: KalmanFilter
    init {
        id = count++
        x = boundingBox?.let { getBoxFeatures(it) }!!
        x = x.append(ArrayRealVector(doubleArrayOf(0.0,0.0,0.0)))
//        Log.d("F", "${F.rowDimension.toString()} X ${F.columnDimension.toString()}")
//        Log.d("Q", "${Q.rowDimension.toString()} X ${Q.columnDimension.toString()}")
//        Log.d("x", "${x.dimension.toString()}")
//        Log.d("P", "${P.rowDimension.toString()} X ${P.columnDimension.toString()}")
//        Log.d("H", "${H.rowDimension.toString()} X ${H.columnDimension.toString()}")
//        Log.d("R", "${R.rowDimension.toString()} X ${R.columnDimension.toString()}")
//        Log.d("F", F.toString())
//        Log.d("Q", Q.toString())
//        Log.d("xVal", x.toString())
//        Log.d("P", P.toString())
//        Log.d("H", H.toString())
//        Log.d("R", R.toString())
        processModel = DefaultProcessModel(F,null,Q,x,P)
        measurementModel = DefaultMeasurementModel(H, R)
        kalmanFilter = KalmanFilter(processModel,measurementModel)
//        kalmanFilter = DefaultK

        val h = this.getState()
//        Log.d("BOX AFTER",kalmanFilter.stateEstimation.contentToString() )
    }


    fun update(boundingBox: FloatArray) {
        val x_new = getBoxFeatures(boundingBox)
        this.timeSinceUpdate = 0.0f
        this.history.clear()
        this.hits++
        this.hitStreak++
        this.kalmanFilter.correct(x_new)
    }

    fun predict(): FloatArray {
        if ((kalmanFilter.stateEstimation[6] + kalmanFilter.stateEstimation[2]) <= 0) {
            kalmanFilter.stateEstimation[6] *= 0.0
        }

        kalmanFilter.predict()

        this.age++

        if (this.timeSinceUpdate > 0) {
            this.hitStreak = 0
        }

        this.timeSinceUpdate++

        this.history.add(convertBBox(kalmanFilter.stateEstimation))

        return this.history.last()
    }

    fun getState() : FloatArray{
        return convertBBox(this.kalmanFilter.stateEstimation)
    }
}