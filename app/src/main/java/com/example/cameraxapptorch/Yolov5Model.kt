package com.example.cameraxapptorch

import java.nio.MappedByteBuffer

object Yolov5Model {
    private lateinit var mappedByteBuffer: MappedByteBuffer
    private var dequantizeFactor = 0.02953994646668434
    private var dequantizeBias = 2

    fun setMappedByteBuffer(buffer: MappedByteBuffer) {
        mappedByteBuffer = buffer
    }

    fun getMappedByteBuffer(): MappedByteBuffer {
        return mappedByteBuffer
    }

    fun setDequantizeFactorAndBias(_dequantizeFactor: Double, _dequantizeBias: Int) {
        dequantizeFactor = _dequantizeFactor
        dequantizeBias = _dequantizeBias
    }

    fun getDequantizeFactor(): Double {
        return dequantizeFactor
    }

    fun getDequantizeBias(): Int {
        return dequantizeBias
    }
}