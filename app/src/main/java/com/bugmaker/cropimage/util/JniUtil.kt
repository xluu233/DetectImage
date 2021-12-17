package com.bugmaker.cropimage.util

import com.bugmaker.cropimage.Point


/**
 * @ClassName JniUtil
 * @Description TODO
 * @Author AlexLu_1406496344@qq.com
 * @Date 2021/12/17 15:44
 */
object JniUtil {

    /**
     * TODO 初始化
     *
     * @param modePath 模型路径
     * @return
     */
    external fun init(modePath:String):Boolean

    external fun detectImage(byteArray: ByteArray,width:Int,height:Int):Array<Point>?

}