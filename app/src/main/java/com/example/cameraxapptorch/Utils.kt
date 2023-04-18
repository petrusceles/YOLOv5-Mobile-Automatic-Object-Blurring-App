package com.example.cameraxapptorch

import kotlin.math.max
import kotlin.math.min

class Utils {
    companion object {
        fun bb_iou(bb_test: FloatArray, bb_gt: FloatArray): Double {
            val xx1 = max(bb_test[0], bb_gt[0])
            val yy1 = max(bb_test[1], bb_gt[1])
            val xx2 = min(bb_test[2], bb_gt[2])
            val yy2 = min(bb_test[3], bb_gt[3])
            val w = max(0.0, (xx2 - xx1).toDouble())
            val h = max(0.0, (yy2 - yy1).toDouble())
            val wh = w * h
            return wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                    (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        }
    }
}