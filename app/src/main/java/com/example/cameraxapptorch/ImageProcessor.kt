package com.example.cameraxapptorch

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class ImageProcessor {
    companion object {
        fun resizeOnly(bitmap: Bitmap, width: Int, height: Int): Bitmap {
            return paddingImage(resizeImage(bitmap,width,height),width,height)
        }
        fun resizeImage(bitmap: Bitmap, targetWidth: Int, targetHeight: Int):Bitmap {
            val originalWidth = bitmap.width
            val originalHeight = bitmap.height

            val scaleFactor = Math.max(
                originalWidth.toFloat() / targetWidth,
                originalHeight.toFloat() / targetHeight
            )

            val newWidth = (originalWidth / scaleFactor).toInt()
            val newHeight = (originalHeight / scaleFactor).toInt()

            return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
        }
        fun paddingImage(bitmap: Bitmap, targetWidth: Int, targetHeight: Int) :Bitmap {
            val leftPadding = (targetWidth - bitmap.width) / 2
            val topPadding = (targetHeight - bitmap.height) / 2
            val outputBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(outputBitmap)
            canvas.drawColor(Color.rgb(114,114,114))
            canvas.drawBitmap(bitmap, leftPadding.toFloat(), topPadding.toFloat(), null)
            return outputBitmap
        }
        fun resizeAndGrayScale(bitmap: Bitmap, width: Int, height: Int): Bitmap {
            return paddingImage(convertToGrayscale(resizeImage(bitmap,width,height)), width, height)
        }
        fun convertToGrayscale(bitmap: Bitmap): Bitmap {
            val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            val colorMatrix = ColorMatrix()
            colorMatrix.setSaturation(0f)
            val paint = Paint()
            paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
            val canvas = Canvas(grayscaleBitmap)
            canvas.drawBitmap(bitmap, 0f, 0f, paint)
            return grayscaleBitmap
        }
        fun resizeAndHisteq(bitmap: Bitmap, width: Int, height: Int): Bitmap {
            return paddingImage(histogramEqualization(resizeImage(bitmap,width,height)), width, height)
        }
        fun histogramEqualization(bitmap: Bitmap): Bitmap {
            val histogram = IntArray(256)
            val equalizedHistogram = IntArray(256)
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            for (pixel in pixels) {
                val grayscale = Color.red(pixel)
                histogram[grayscale]++
            }
            val totalPixels = bitmap.width * bitmap.height
            var sum = 0
            for (i in 0 until 256) {
                sum += histogram[i]
                equalizedHistogram[i] = (sum.toFloat() / totalPixels * 255).toInt()
            }
            val equalizedBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            for (i in 0 until pixels.size) {
                val grayscale = Color.red(pixels[i])
                val equalizedValue = equalizedHistogram[grayscale]
                pixels[i] = Color.rgb(equalizedValue, equalizedValue, equalizedValue)
            }
            equalizedBitmap.setPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            return equalizedBitmap
        }
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