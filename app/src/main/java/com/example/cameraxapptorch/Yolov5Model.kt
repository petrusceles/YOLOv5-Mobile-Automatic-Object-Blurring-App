package com.example.cameraxapptorch

import java.nio.MappedByteBuffer

object Yolov5Model {
    private lateinit var mappedByteBuffer: MappedByteBuffer
    private var confThreshold = 0.5f
    private var iouThreshold = 0.5f
    private var folderPrefix = ""
    private var folderMain = ""
    private var isTracking = false
    private var isSaveUntracked = false

    fun setMappedByteBuffer(buffer: MappedByteBuffer) {
        mappedByteBuffer = buffer
    }

    fun getMappedByteBuffer(): MappedByteBuffer {
        return mappedByteBuffer
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

    fun setConfThreshold(conf_threshold: Float) {
        confThreshold = conf_threshold
    }

    fun setIouThreshold(iou_threshold: Float) {
        iouThreshold = iou_threshold
    }

    fun setIsSaveUntracked(is_save_untracked: Boolean) {
        isSaveUntracked = is_save_untracked
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

    fun getConfThreshold(): Float {
        return confThreshold
    }

    fun getIouThreshold(): Float {
        return iouThreshold
    }

    fun getIsSaveUntracked() :Boolean {
        return isSaveUntracked
    }
}