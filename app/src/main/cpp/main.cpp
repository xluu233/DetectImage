//
// Created by luhon on 2021/12/17.
//

#include "main.h"
#include "detect/PageDet.h"
#include <jni.h>
#include <imgproc/types_c.h>

string jstring2string(JNIEnv *env,jstring js){
    const char* cppMsg=env->GetStringUTFChars(js, JNI_FALSE);
    env->ReleaseStringUTFChars(js, cppMsg);
    string str = cppMsg;
    return str;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_bugmaker_cropimage_util_JniUtil_init(JNIEnv *env, jobject thiz, jstring mode_path) {
    if (PageDet::detector != nullptr) {
        delete PageDet::detector;
        PageDet::detector = nullptr;
    }
    if (PageDet::detector == nullptr) {
        string modelPath = jstring2string(env,mode_path);
        LOGD("model path:%s", modelPath.c_str());
        PageDet::detector = new PageDet(modelPath, true);
        return true;
    }
    return false;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_bugmaker_cropimage_util_JniUtil_detectImage(JNIEnv *env, jobject thiz,
                                                     jbyteArray byte_array, jint width,
                                                     jint height) {

    jbyte *imageDate = env->GetByteArrayElements(byte_array, nullptr);
    auto *dataTemp = (unsigned char *) imageDate;
    cv::Mat tempMat(height, width, CV_8UC4, dataTemp);
    //色彩转换
    cv::Mat oriMat;
    cv::cvtColor(tempMat, oriMat, CV_RGBA2RGB);
    tempMat.release();

    if (oriMat.channels() != 3) {
        LOGE("input image format channels != 3");
    }
    //分辨率512
    cv::Mat biMat = PageDet::detector->infer(oriMat);
    oriMat.release();
    env->ReleaseByteArrayElements(byte_array, imageDate, 0);
    std::vector<cv::Point2d> img_corner = ProcessEdgeImageV2(biMat);

    double ratio_h = ((double)height)/512.0;
    double ratio_w = ((double)width)/512.0;

    LOGD("img_corner size: %lu",img_corner.size());
    for (auto& point : img_corner) {
        point.x = (point.x*ratio_w) < 0 ? 0 : (point.x*ratio_w);
        point.y = (point.y*ratio_h) < 0 ? 0 : (point.y*ratio_h);

        point.x = (point.x*ratio_w) > width ? width : (point.x*ratio_w);
        point.y = (point.y*ratio_h) > height ? height : (point.y*ratio_h);

        LOGD("point: %d %d",(int)(point.x),(int)(point.y));
    }

    //android/graphics/Point
    //com/bugmaker/cropimage/Point
    jclass pointClass = env->FindClass("com/bugmaker/cropimage/Point");
    if (pointClass == nullptr){
        LOGD("Point class is null");
        return nullptr;
    }
    jmethodID init = env->GetMethodID(pointClass, "<init>", "(II)V");
    jobjectArray pointArray = env->NewObjectArray(img_corner.size(),pointClass, NULL);

    for (int i = 0; i < img_corner.size(); ++i) {
        env->SetObjectArrayElement(pointArray, i,env->NewObject(pointClass, init, (int)img_corner[i].x,(int)img_corner[i].y));
    }

    env->DeleteLocalRef(pointClass);
    return pointArray;
}