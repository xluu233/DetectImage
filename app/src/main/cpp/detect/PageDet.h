//
// Created by alvin on 2021/11/25.
//

#ifndef MNN_DEMO_PAGEDET_H
#define MNN_DEMO_PAGEDET_H

#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/bitmap.h>
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


#ifndef LOG_TAG
#define LOG_TAG "ALVIN_MNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif

using namespace MNN;
using namespace std;
using namespace cv;

//typedef struct PageInfo_ {
//    std::vector<cv::Point2d> corner;
//    float score=1.0;
//} PageInfo;


class PageDet {
public:
    PageDet(const std::string &mnn_path, bool useGPU);
    ~PageDet();

    cv::Mat infer(const cv::Mat&);

private:

    std::shared_ptr<MNN::Interpreter> PageDet_interpreter;
    MNN::Session *PageDet_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    MNN::Tensor *output_tensor = nullptr;

//    int in_n = 1;
//    int in_c = 3;
//    int in_w = 512;
//    int in_h = 512;

public:
    static PageDet *detector;
    static bool hasGPU;
    static bool toUseGPU;
};

typedef struct vLine_s_ {
    cv::Vec4i cdline;
    double b = 0.;
    double conf = 0.;
}vLine_s;

typedef struct non_vLine_s_ {
    cv::Vec4i cdline;
    double k = 0.;
    double b = 0.;
    double conf = 0.;
}non_vLine_s;

typedef struct vLineConf_value_ {
    double v_x_conf = 0.;
    std::vector<vLine_s> v_main_line;
}vLineConf_value;

typedef struct nonvLineConf_value_ {
    double nv_x_conf = 0.;
    std::vector<non_vLine_s> nv_main_line;
}nonvLineConf_value;

typedef struct kandb_ {
    double k = 0.;
    double b = 0.;
}kandb;

std::vector<cv::Point2d> ProcessEdgeImageV2(const cv::Mat &);


#endif //MNN_DEMO_PAGEDET_H