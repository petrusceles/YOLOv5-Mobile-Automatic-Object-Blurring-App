package com.example.cameraxapptorch

import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.RealVector
import org.apache.commons.math3.filter.MeasurementModel
import org.apache.commons.math3.filter.ProcessModel
import org.apache.commons.math3.filter.DefaultMeasurementModel
import org.apache.commons.math3.filter.DefaultProcessModel
import org.apache.commons.math3.filter.KalmanFilter

class KalmanBoxTracker(
    accelerationNoise: Double = 0.1,
    measurementNoise: Double = 10.0,
    boundingBox: FloatArray?
) {
    private val numStates = 7
    private val numMeasurements = 4

    private val F: RealMatrix = Array2DRowRealMatrix(arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        doubleArrayOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
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
        setEntry(4, 4, 1000.0)
        setEntry(5, 5, 1000.0)
        setEntry(6, 6, 1000.0)
        scalarMultiply(10.0)
    }

    private var x: RealVector

    private val Q: RealMatrix = Array2DRowRealMatrix(7, 7).apply {
        setEntry(6, 6, 0.01)
        setSubMatrix(Array2DRowRealMatrix(Array(7) { i -> doubleArrayOf(0.0) }).scalarMultiply(0.01).data, 0, 0)
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

    fun convertBBox(bBox: DoubleArray): FloatArray {
        val xCenter = bBox[0]
        val yCenter = bBox[1]
        val widthHeight = bBox[2] / 2.0 * bBox[3]
        val x1 = xCenter - widthHeight
        val y1 = yCenter - widthHeight / bBox[3]
        val x2 = xCenter + widthHeight
        val y2 = yCenter + widthHeight / bBox[3]
        return floatArrayOf(x1.toFloat(), y1.toFloat(), x2.toFloat(), y2.toFloat())
    }

    private var processModel: DefaultProcessModel

    private var measurementModel: MeasurementModel

    private var timeSinceUpdate = 0.0f

    private var history = mutableListOf<FloatArray>()

    private var hits :Int = 0

    private var hitStreak: Int = 0

    private var age :Int = 0

    companion object {
        private var count = 0
    }
    private var id: Int = 0

    private var kalmanFilter: KalmanFilter
    init {
        id = count++
        x = boundingBox?.let { getBoxFeatures(it) }!!
        processModel = DefaultProcessModel(F,null,Q,x,P)
        measurementModel = DefaultMeasurementModel(H, R)
        kalmanFilter = KalmanFilter(processModel,measurementModel)
    }


    private fun update(boundingBox: FloatArray) {
        val x_new = getBoxFeatures(boundingBox)
        this.timeSinceUpdate = 0.0f
        this.history.clear()
        this.hits++
        this.hitStreak++
        this.kalmanFilter.correct(x_new)
    }

    private fun predict(): FloatArray {
        if ((kalmanFilter.stateEstimation[6] + kalmanFilter.stateEstimation[2]) >= 0) {
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

    private fun getState() : FloatArray{
        return convertBBox(this.kalmanFilter.stateEstimation)
    }

}