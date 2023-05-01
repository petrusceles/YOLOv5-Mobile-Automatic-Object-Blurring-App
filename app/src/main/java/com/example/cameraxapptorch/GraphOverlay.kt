package com.example.cameraxapptorch

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.View

class GraphOverlay(context: Context, attrs: AttributeSet) : View(context, attrs) {

//    private val paint = Paint().apply {
//        color = Color.GREEN
//        style = Paint.Style.STROKE
//        strokeWidth = 4f
//    }

    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL


//        maskFilter = BlurMaskFilter(180f, BlurMaskFilter.Blur.NORMAL)
    }

    private val letterPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        textSize = 50f
        textAlign = Paint.Align.CENTER
    }

    private var rect: MutableList<FloatArray>? = null

    fun setRect(rect: MutableList<FloatArray>) {
        this.rect = rect
        invalidate()
    }

    fun clearRect() {
        this.rect = null
        invalidate()
    }

    @SuppressLint("DrawAllocation")
    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        rect?.let {
            for (single in it) {
                canvas?.drawRoundRect(RectF(single[0],single[1],single[2],single[3]),25f,25f,paint)
                Log.d("BOX SIZE BEFORE", single.size.toString())
                if (single.size == 5) {
                    Log.d("BOX SIZE", single.size.toString())
                    val xC = (single[0] + single[2])/2
                    val yC = (single[1] + single[3])/2
                    canvas?.drawText(single[4].toInt().toString(),single[0],single[1],letterPaint)
                }
            }
        }
    }
}