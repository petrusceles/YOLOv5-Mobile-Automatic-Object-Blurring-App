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
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.util.Log
import com.example.cameraxapptorch.databinding.ActivityMainBinding
const val PICK_PDF_FILE = 2
class MenuActivity : ComponentActivity() {
    private lateinit var viewBinding: ActivityMenuBinding
    private val PICK_TFLITE_FILE_REQUEST = "pickTfliteFileRequest"
    private var fileExtension = ""
    lateinit var tfliteModel: MappedByteBuffer
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMenuBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        viewBinding.chooseFileButton.setOnClickListener {
            Intent(Intent.ACTION_GET_CONTENT).also {
                it.type = "*/*"
                startActivityForResult(it,0)
            }
        }
        viewBinding.startButton.setOnClickListener {
            Log.d("fileExtension",fileExtension)
            if (fileExtension == "tflite") {
                try {
                    Yolov5Model.setMappedByteBuffer(tfliteModel)
                    val intent = Intent(this, MainActivity::class.java)
                    startActivity(intent)
                } catch (error: Error) {
                    showToast("Model Error")
                }
            } else {
                showToast("Model Error")
            }
        }

        viewBinding.dequantizeValueSetButton.setOnClickListener {
            if (viewBinding.dequantizeBiasText.text.toString().toDoubleOrNull() == null  || viewBinding.dequantizeFactorText.text.toString().toDoubleOrNull() == null) {
                showToast("Dequantize value invalid")
            } else {
                val dequantizeBias = viewBinding.dequantizeBiasText.text.toString().toInt()
                val dequantizeFactor = viewBinding.dequantizeFactorText.text.toString().toDouble()
                Yolov5Model.setDequantizeFactorAndBias(dequantizeFactor,dequantizeBias)
                Log.d("DEQUANTIZE BIAS", dequantizeBias.toString())
                Log.d("DEQUANTIZE FACTOR", dequantizeFactor.toString())
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode ==Activity.RESULT_OK && requestCode==0) {
            val selectedFileUri: Uri? = data?.data
            if (selectedFileUri != null) {
                fileExtension = getFileName(selectedFileUri)!!
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



//    private fun checkPermissionAndOpenFilePicker() {
//        val permission = Manifest.permission.READ_EXTERNAL_STORAGE
//        Log.d(PICK_TFLITE_FILE_REQUEST,permission)
//        if (ContextCompat.checkSelfPermission(this, permission) ==
//            PackageManager.PERMISSION_GRANTED
//        ) {
//            openFilePicker()
//        } else {
//            if (ActivityCompat.shouldShowRequestPermissionRationale(this, permission)) {
//                showToast("Storage permission is required to upload a file")
//            }
//        }
//    }
//
//
//    private val pickFileLauncher = registerForActivityResult(
//        ActivityResultContracts.StartActivityForResult()
//    ) { result ->
//        if (result.resultCode == Activity.RESULT_OK) {
//            val data: Intent? = result.data
//            val selectedFileUri: Uri? = data?.data
//            if (selectedFileUri != null) {
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


    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}