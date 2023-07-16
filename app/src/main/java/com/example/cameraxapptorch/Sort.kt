package com.example.cameraxapptorch

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlin.math.max
import kotlin.math.min

class Sort(module: PyObject) {

    private var objectTrackers: MutableList<KalmanBoxTracker> = mutableListOf()
    private var frameCount: Int = 0
    private val module = module

    fun updateSort(detections: List<FloatArray> = mutableListOf()): MutableList<FloatArray> {
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
        val associations = associateDetectionsToTrackers(detections, tracks, 0.1f)
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
            i -= 1
            if (tracker.timeSinceUpdate > Yolov5Model.getMaxTrackerAge()) {
                this.objectTrackers.removeAt(i)
            } else {
                if (tracker.hitStreak > 1 || frameCount <= 1 || tracker.age == 0) {
                    returnedBBox.add(bbox.plus((tracker.id + 1).toFloat()))
                }
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
            return Triple(mutableListOf(), unmatchedDetections, mutableListOf())
        }

        val iouMatrix = computeIoUMatrix(detections, trackers)

        val allEmpty = iouMatrix.any { it.isEmpty() }

        var matchedIndices = mutableListOf<IntArray>()
        if (iouMatrix.isNotEmpty() && !allEmpty) {
            val numRows = iouMatrix.size
            val numCols = iouMatrix[0].size
            val thresholdMatrix = Array(numRows) { IntArray(numCols) }

            for (i in 0 until numRows) {
                for (j in 0 until numCols) {
                    if (iouMatrix[i][j] > iouThreshold) {
                        thresholdMatrix[i][j] = 1
                    } else {
                        thresholdMatrix[i][j] = 0
                    }
                }
            }

            val a = findMaxRowAndColSum(thresholdMatrix)

            if (a.first == 1 && a.second == 1) {
                matchedIndices = findOnes(thresholdMatrix)
            } else {
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
//        for (match in matchedIndices) {
//            Log.d("MATCH", match.contentToString())
//        }

        val matches = mutableListOf<IntArray>()

        for (it in matchedIndices) {
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

                val unionArea = trackerArea + detectionArea - interArea

                val iou = interArea / unionArea
                iouMatrix[i][j] = iou
            }
        }
        return iouMatrix
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
}