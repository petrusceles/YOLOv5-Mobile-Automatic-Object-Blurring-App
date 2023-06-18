package com.example.cameraxapptorch

import java.nio.MappedByteBuffer

object Yolov5Model {
    private lateinit var mappedByteBuffer: MappedByteBuffer
    private var dequantizeFactor = 0.02953994646668434
    private var dequantizeBias = 2
    private var folderPrefix = ""
    private var folderMain = ""
    private var isTracking = false

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

    fun setFolderPrefix(prefix: String) {
        folderPrefix = prefix
    }

    fun setFolderMain(main: String) {
        folderMain = main
    }

    fun setIsTracking(is_tracking: Boolean) {
        isTracking = is_tracking
    }

    fun getDequantizeFactor(): Double {
        return dequantizeFactor
    }

    fun getDequantizeBias(): Int {
        return dequantizeBias
    }

    fun getFolderPrefix(): String {
        return folderPrefix
    }
    fun getFolderMain(): String {
        return folderMain
    }

    fun getIsTracking(): Boolean{
        return isTracking
    }
}