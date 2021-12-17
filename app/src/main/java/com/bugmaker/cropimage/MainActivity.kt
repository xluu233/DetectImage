package com.bugmaker.cropimage

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Point
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.annotation.Nullable
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.os.EnvironmentCompat
import androidx.lifecycle.lifecycleScope
import com.bugmaker.cropimage.databinding.ActivityMainBinding
import com.bugmaker.cropimage.util.AssertMangerUtil
import com.bugmaker.cropimage.util.BitmapUtils
import com.bugmaker.cropimage.util.JniUtil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    // 申请相机权限的requestCode
    private val PERMISSION_CAMERA_REQUEST_CODE = 0x00000012
    //用于保存拍照图片的uri
    private var mCameraUri: Uri? = null
    // 用于保存图片的文件路径，Android 10以下使用图片路径访问图片
    private var mCameraImagePath: String? = null
    // 是否是Android 10以上手机
    private val isAndroidQ = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q

    private val REQUEST_CODE = 110

    private var bitmap: Bitmap? = null

    private lateinit var binding: ActivityMainBinding

    init {
        System.loadLibrary("detect-opencv")
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        initView()
    }

    override fun onResume() {
        super.onResume()
        initModel()
    }

    private fun initView() {
        //选择照片，直接打开相机无需请求权限
        binding.talePhoto.setOnClickListener {
            //checkPermissionAndCamera()
            openCamera()
        }

        //获取坐标点
        binding.getPoint.setOnClickListener(){

        }

        //裁剪
        binding.crop.setOnClickListener {

        }

        //识别
        binding.detect.setOnClickListener {
            detectImage(bitmap)
        }

    }

    private fun initModel() {
        val fileName = "page_sfv2_6641_v2_sim.mnn"
        val filePath = this.filesDir.absolutePath
        val file = File(filePath,fileName)
        if (file.exists()){
            JniUtil.init(file.absolutePath)
        }else{
            val result = AssertMangerUtil.copyFileFromAssets(this,fileName,filePath,fileName)
            if (result) initModel()
        }
    }


    /**
     * 检查权限并拍照,调用相机前先检查权限。
     */
    private fun checkPermissionAndCamera() {
        val hasCameraPermission = ContextCompat.checkSelfPermission(
            application,
            Manifest.permission.CAMERA
        )
        if (hasCameraPermission == PackageManager.PERMISSION_GRANTED) {
            //有调起相机拍照。
            openCamera()
        } else {
            //没有权限，申请权限。
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), PERMISSION_CAMERA_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == PERMISSION_CAMERA_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                //允许权限，有调起相机拍照。
                openCamera()
            } else {
                //拒绝权限，弹出提示框。
                Toast.makeText(this, "拍照权限被拒绝", Toast.LENGTH_LONG).show()
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }


    /**
     * 调起相机拍照
     */
    private fun openCamera() {
        val captureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        // 判断是否有相机
        if (captureIntent.resolveActivity(packageManager) != null) {
            var photoFile: File? = null
            var photoUri: Uri? = null
            if (isAndroidQ) {
                // 适配android 10
                photoUri = createImageUri()
            } else {
                try {
                    photoFile = createImageFile()
                } catch (e: IOException) {
                    e.printStackTrace()
                }
                if (photoFile != null) {
                    mCameraImagePath = photoFile.absolutePath
                    photoUri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                        //适配Android 7.0文件权限，通过FileProvider创建一个content类型的Uri
                        FileProvider.getUriForFile(this, "$packageName.fileprovider", photoFile)
                    } else {
                        Uri.fromFile(photoFile)
                    }
                }
            }
            mCameraUri = photoUri
            if (photoUri != null) {
                captureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
                captureIntent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
                startActivityForResult(captureIntent, REQUEST_CODE)
            }
        }
    }

    /**
     * 创建图片地址uri,用于保存拍照后的照片 Android 10以后使用这种方法
     */
    private fun createImageUri(): Uri? {
        val status: String = Environment.getExternalStorageState()
        // 判断是否有SD卡,优先使用SD卡存储,当没有SD卡时使用手机存储
        return if (status == Environment.MEDIA_MOUNTED) {
            contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, ContentValues())
        } else {
            contentResolver.insert(MediaStore.Images.Media.INTERNAL_CONTENT_URI, ContentValues())
        }
    }

    /**
     * 创建保存图片的文件
     */
    @Throws(IOException::class)
    private fun createImageFile(): File? {
        val imageName: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        if (!storageDir?.exists()!!) {
            storageDir.mkdir()
        }
        val tempFile = File(storageDir, imageName)
        return if (!Environment.MEDIA_MOUNTED.equals(EnvironmentCompat.getStorageState(tempFile))) {
            null
        } else tempFile
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, @Nullable data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == REQUEST_CODE) {
            if (isAndroidQ) {
                // Android 10 使用图片uri加载
                bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(mCameraUri!!))
            } else {
                // 使用图片路径加载
                bitmap = BitmapFactory.decodeFile(mCameraImagePath)
            }
            Log.d("TAG", "onActivityResult: ${bitmap?.width},${bitmap?.height}")
            detectImage(bitmap)
        }

    }

    private fun detectImage(bitmap: Bitmap?) = lifecycleScope.launch(Dispatchers.IO){
        if (bitmap == null) return@launch

        withContext(Dispatchers.Main){
            binding.cropImageview.setImageToCrop(bitmap)
        }

        val bitmapHeight = bitmap.height
        val bitmapWidth = bitmap.width
        val byteArray = BitmapUtils.convertToByteArray(bitmap) ?: return@launch

        val result = JniUtil.detectImage(byteArray,bitmapWidth,bitmapHeight)
        result?.forEach {
            Log.d("detectImage",it.toString())
        }

        withContext(Dispatchers.Main){
            binding.cropImageview.apply {
                //cropPoints = result
            }
        }

    }


}