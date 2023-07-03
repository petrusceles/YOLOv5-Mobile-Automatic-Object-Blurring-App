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

        fun scaleAndHisteq(bitmap: Bitmap, width: Int, height: Int): Bitmap {
            return histogramEqualization(Bitmap.createScaledBitmap(bitmap,width,height,false))
        }

        fun scaleAndGrayScale(bitmap: Bitmap, width: Int, height: Int): Bitmap {
            return convertToGrayscale(Bitmap.createScaledBitmap(bitmap,width,height,false))
        }

        fun scaleOnly(bitmap: Bitmap, width: Int, height: Int): Bitmap {
            return Bitmap.createScaledBitmap(bitmap,width,height,false)
        }

        fun convertToGrayscale(bitmap: Bitmap): Bitmap {
            // Create a mutable bitmap with the same width and height as the original bitmap
            val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)

            // Create a color matrix for grayscale conversion
            val colorMatrix = ColorMatrix()
            colorMatrix.setSaturation(0f)

            // Create a paint object with the color matrix
            val paint = Paint()
            paint.colorFilter = ColorMatrixColorFilter(colorMatrix)

            // Create a canvas with the grayscale bitmap
            val canvas = android.graphics.Canvas(grayscaleBitmap)

            // Draw the original bitmap on the canvas with the paint
            canvas.drawBitmap(bitmap, 0f, 0f, paint)

            return grayscaleBitmap
        }

        fun histogramEqualization(bitmap: Bitmap): Bitmap {
            // Calculate the histogram of the grayscale image
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

            // Create a new bitmap for the equalized image
            val equalizedBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)

            // Apply the equalization to each pixel in the grayscale image
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