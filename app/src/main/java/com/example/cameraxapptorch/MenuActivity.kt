package com.example.cameraxapptorch

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.cameraxapptorch.databinding.ActivityMenuBinding
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentResolver
import android.database.Cursor
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.util.Log
import androidx.core.widget.addTextChangedListener
import androidx.core.widget.doOnTextChanged
import com.example.cameraxapptorch.databinding.ActivityMainBinding
@Suppress("DEPRECATION")
class MenuActivity : ComponentActivity() {
    private lateinit var viewBinding: ActivityMenuBinding
    private var fileExtension = ""
    private lateinit var tfliteModel: MappedByteBuffer

    private var activityResultLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        if (it.resultCode == Activity.RESULT_OK && it.data != null) {
            val selectedFileUri: Uri? = it.data?.data
            if (selectedFileUri != null) {
                fileExtension = getFileName(selectedFileUri)!!
                getFilenameWithoutExtensionFromUri(selectedFileUri)?.let {
                    Yolov5Model.setFolderMain(
                        it
                    )
                }
                val fileDescriptor = contentResolver.openFileDescriptor(selectedFileUri, "r")
                if (fileDescriptor != null) {
                    val fileInputStream = FileInputStream(fileDescriptor.fileDescriptor)
                    val fileChannel = fileInputStream.channel
                    val startOffset = 0L
                    val declaredLength = fileDescriptor.statSize
                    tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
                    showToast("File uploaded successfully!")
                } else {
                    showToast("Error opening the file")
                }
            } else {
                showToast("File selection canceled")
            }
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMenuBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        viewBinding.chooseFileButton.setOnClickListener {
            Intent(Intent.ACTION_GET_CONTENT).also {
                it.type = "*/*"
                activityResultLauncher.launch(it)
            }
        }
        viewBinding.startButton.setOnClickListener {
            Log.d("fileExtension",fileExtension)
            if (fileExtension == "tflite") {
                try {
                    Yolov5Model.setMappedByteBuffer(tfliteModel)
                    val intent = Intent(this, MainActivity::class.java)
                    startActivity(intent)
                } catch (error: Error) { showToast("Model Error") }
            } else { showToast("Model Error") }
        }
        viewBinding.setThresholdValue.setOnClickListener {
            try {
                Yolov5Model.setConfThreshold(viewBinding.confThresh.text.toString().toFloat())
                Yolov5Model.setIouThreshold(viewBinding.iouThresh.text.toString().toFloat())
                Yolov5Model.setMaxTrackerAge(viewBinding.maxTrackerAge.text.toString().toInt())
            } catch (err: Error) {
                showToast("Error threshold values")
            }
        }
        viewBinding.folderPrefixText.doOnTextChanged { text, _, _, _ -> Yolov5Model.setFolderPrefix(text.toString()) }
        viewBinding.trackingCheck.setOnCheckedChangeListener { _, isChecked -> Yolov5Model.setIsTracking(isChecked) }
        viewBinding.saveUntrackedCheck.setOnCheckedChangeListener { _, isChecked ->  Yolov5Model.setIsSaveUntracked(isChecked) }
        viewBinding.radioImageProcessing.setOnCheckedChangeListener { _, checkedId ->
            Yolov5Model.setGrayscale(false)
            Yolov5Model.setHisteq(false)
            when (checkedId) {
                viewBinding.histeq.id -> {
                    Yolov5Model.setHisteq(true)
                }
                viewBinding.grayscale.id -> {
                    Yolov5Model.setGrayscale(true)
                }
            }
        }
    }
    private fun getFilenameWithoutExtensionFromUri(uri: Uri): String? {
        var filenameWithoutExtension: String? = null
        val contentResolver: ContentResolver = applicationContext.contentResolver
        val cursor: Cursor? = contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val displayNameIndex: Int = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (displayNameIndex != -1) {
                    val displayName: String = it.getString(displayNameIndex)
                    val dotIndex: Int = displayName.lastIndexOf(".")
                    if (dotIndex != -1) {
                        filenameWithoutExtension = displayName.substring(0, dotIndex)
                    }
                }
            }
        }
        return filenameWithoutExtension
    }
    @SuppressLint("Range")
    private fun getFileName(uri: Uri): String? {
        var fileName: String? = null
        val cursor = contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val displayName =
                    it.getString(it.getColumnIndex(OpenableColumns.DISPLAY_NAME))
                fileName = displayName
            }
        }
        fileName = fileName?.substringAfterLast(".", "")
        return fileName
    }
    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }



//    @SuppressLint("Recycle")
//    @Deprecated("Deprecated in Java")
//    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
//        super.onActivityResult(requestCode, resultCode, data)
//        if (resultCode ==Activity.RESULT_OK && requestCode==0) {
//            val selectedFileUri: Uri? = data?.data
//            if (selectedFileUri != null) {
//                fileExtension = getFileName(selectedFileUri)!!
//                getFilenameWithoutExtensionFromUri(selectedFileUri)?.let {
//                    Yolov5Model.setFolderMain(
//                        it
//                    )
//                }
//                val fileDescriptor = contentResolver.openFileDescriptor(selectedFileUri, "r")
//                if (fileDescriptor != null) {
//                    val fileInputStream = FileInputStream(fileDescriptor.fileDescriptor)
//                    val fileChannel = fileInputStream.channel
//                    val startOffset = 0L
//                    val declaredLength = fileDescriptor.statSize
//                    tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
//                    showToast("File uploaded successfully!")
//                } else {
//                    showToast("Error opening the file")
//                }
//            } else {
//                showToast("File selection canceled")
//            }
//        }
//    }
}