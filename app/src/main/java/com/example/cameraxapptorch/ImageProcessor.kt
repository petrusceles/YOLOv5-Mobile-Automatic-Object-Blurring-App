package com.example.cameraxapptorch

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class ImageProcessor {
    companion object {

        fun imageProxyToBitmap(imageProxy: ImageProxy, rotation: Int): Bitmap {
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
    }
}