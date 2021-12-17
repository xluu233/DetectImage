//
// Created by alvin on 2021/11/25.
//

#include <opencv2/imgproc/types_c.h>
#include "PageDet.h"
//#include "omp.h"
#define TAG "pageInfer"


bool PageDet::hasGPU = false;
bool PageDet::toUseGPU = false;
PageDet *PageDet::detector = nullptr;

PageDet::PageDet(const std::string &mnn_path, bool useGPU) {
    toUseGPU = hasGPU && useGPU;

    PageDet_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = 4;
//    config.type = MNN_FORWARD_AUTO;
    if (useGPU) {
        config.type = MNN_FORWARD_OPENCL;
    }
    else {
        config.type = MNN_FORWARD_AUTO;
    }
    config.backupType = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;  // 内存
    backendConfig.power = MNN::BackendConfig::Power_Low;  // 功耗
    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;  // 精度
    config.backendConfig = &backendConfig;

    PageDet_session = PageDet_interpreter->createSession(config);
    if (nullptr == PageDet_session){
        LOGW("other dimension type");
    }
    input_tensor = PageDet_interpreter->getSessionInput(PageDet_session, nullptr);
    output_tensor = PageDet_interpreter->getSessionOutput(PageDet_session, nullptr);
}

PageDet::~PageDet() {
    PageDet_interpreter->releaseModel();
    PageDet_interpreter->releaseSession(PageDet_session);
}

cv::Mat PageDet::infer(const cv::Mat &raw_img) {

    std::vector<int> shape{ 1, 3, 512, 512 };
    PageDet_interpreter->resizeTensor(input_tensor, shape);
    PageDet_interpreter->resizeSession(PageDet_session);
    int width = input_tensor->width();
    int height = input_tensor->height();
    int channel = input_tensor->channel();
    int size = width * height;

    //cv::cvtColor(raw_img, raw_img, cv::COLOR_BGR2RGB);
    int img_width = raw_img.cols;
    int img_height = raw_img.rows;

    cv::Mat img_resized;
    cv::resize(raw_img, img_resized, cv::Size(512, 512));

    auto nchwTensor = new Tensor(input_tensor, Tensor::CAFFE);

    float mean[] = { 0.485, 0.456, 0.406 };
    float stddev[] = { 0.229, 0.224, 0.225 };

    // convert nhwc layout to nchw
    for (size_t i = 0; i < channel; i++) {
        for (size_t j = 0; j < size; j++) {
            float value = *(img_resized.data + j * channel + i);
            nchwTensor->host<float>()[size * i + j] = (value / 255. - mean[i]) / stddev[i];
        }
    }

    input_tensor->copyFromHostTensor(nchwTensor);

    // run network
    PageDet_interpreter->runSession(PageDet_session);

    cv::Mat map(512, 512, CV_8UC1,cv::Scalar(0));
    auto values = output_tensor->host<int>();
    for (size_t i = 0; i < 512; i++) {
        auto* data = map.ptr<uchar>(i);
        for (size_t j = 0; j < 512; j++) {
            int cls_id = values[i * 512 + j];
            if (cls_id > 0) {
                data[j] = 255;
            }
        }
    }
    cv::resize(map, map, cv::Size(512, 512));
    img_resized.release();
    delete nchwTensor;
    return map;
}


static void fillUPdefault(std::vector<std::vector<double>> &m_lines, const int ort) {

    if (ort == 0) {
        m_lines[0].push_back(0);
        m_lines[0].push_back(0);
        m_lines[1].push_back(0);
        m_lines[1].push_back(511);
    }
    else if (ort == 1) {
        m_lines[0].push_back(511);
        m_lines[0].push_back(511);
        m_lines[1].push_back(0);
        m_lines[1].push_back(511);
    }
    else if (ort == 2) {
        m_lines[0].push_back(0);
        m_lines[0].push_back(511);
        m_lines[1].push_back(0);
        m_lines[1].push_back(0);
    }
    else {
        m_lines[0].push_back(0);
        m_lines[0].push_back(511);
        m_lines[1].push_back(511);
        m_lines[1].push_back(511);
    }

}


static bool fillUP(std::vector<std::vector<cv::Vec4i>> &all_points, const int &theW) {

    cv::Vec4i temp;
    if (all_points[0].empty()) {
        temp[0] = 0;
        temp[1] = 0;
        temp[2] = 0;
        temp[3] = theW;
        all_points[0].push_back(temp);
    }
    if (all_points[1].empty()) {
        temp[0] = theW;
        temp[1] = 0;
        temp[2] = theW;
        temp[3] = theW;
        all_points[1].push_back(temp);
    }
    if (all_points[2].empty()) {
        temp[0] = 0;
        temp[1] = 0;
        temp[2] = theW;
        temp[3] = 0;
        all_points[2].push_back(temp);
    }
    if (all_points[3].empty()) {
        temp[0] = 0;
        temp[1] = theW;
        temp[2] = theW;
        temp[3] = theW;
        all_points[3].push_back(temp);
    }

    return true;
}


static double confidence(const double &px0, const double &py0, const double &px1, const double &py1) {
    return sqrt(pow(px0 - px1, 2) + pow(py0 - py1, 2));
}


static vLineConf_value vLineConf(const std::vector<vLine_s> &vLine, const std::vector<double> &vLine_x) {
    int s = 1;
    int length = int(vLine_x.size());
    std::vector<std::vector<vLine_s>> split_V;
    int flag = 0;
    while (s < length)
    {
        std::vector<vLine_s> temp;
        temp.push_back(vLine[s - 1]);
        while (abs(vLine_x[s - 1] - vLine_x[s]) <= 2.)
        {
            temp.push_back(vLine[s]);
            s++;
            if (s == length) {
                flag = 1;
                split_V.push_back(temp);
                break;
            }
        }
        if (s != length) {
            split_V.push_back(temp);
        }
        s++;
    }
    if (flag == 0) {
        std::vector<vLine_s> sv_temp;
        sv_temp.push_back(vLine.back());
        split_V.push_back(sv_temp);
    }
    std::vector<double> v_confidence;
    for (auto & sv : split_V) {
        double conf = 0.;
        for (auto & ss : sv) {
            conf += ss.conf;
        }
        v_confidence.push_back(conf);
    }
    auto biggest = std::max_element(std::begin(v_confidence), std::end(v_confidence));
    double max_conf = *biggest;
    auto big_index = std::distance(std::begin(v_confidence), biggest);

    vLineConf_value result;
    result.v_x_conf = max_conf;
    result.v_main_line = split_V[big_index];

    return result;
}


static bool cmp(non_vLine_s &currentl, non_vLine_s &nextl) {
    if (currentl.k != nextl.k) return currentl.k < nextl.k;
    else return currentl.k > nextl.k;
}


static double non_vLineConf(const std::vector<non_vLine_s> &snvLine) {
    int s = 1;
    int snLength = snvLine.size();
    std::vector<std::vector<non_vLine_s>> split_nV;
    int flag = 0;
    while (s < snLength) {
        std::vector<non_vLine_s> stemp = { snvLine[s - 1] };
        while (std::abs(snvLine[s - 1].b - snvLine[s].b) <= 50.) {
            stemp.push_back(snvLine[s]);
            s++;
            if (s == snLength) {
                flag = 1;
                split_nV.push_back(stemp);
                break;
            }
        }
        if (s < snLength) {
            if (std::abs(snvLine[s - 1].b - snvLine[s].b) > 50.) {
                split_nV.push_back(stemp);
            }
        }
        s++;
    }
    if (flag == 0) {
        std::vector<non_vLine_s> lasstemp = { snvLine.back() };
        split_nV.push_back(lasstemp);
    }
    std::vector<double> v_confidence;
    for (auto & sv : split_nV) {
        double conf = 0.;
        //std::vector<std::vector<double>> center_x;
        for (auto & ss : sv) {
            conf += ss.conf;
        }
        v_confidence.push_back(conf);
    }
    double nv_x_conf = *std::max_element(std::begin(v_confidence), std::end(v_confidence));

    return nv_x_conf;
}


static nonvLineConf_value nonVlConf(std::vector<non_vLine_s> &non_vLine) {

    sort(non_vLine.begin(), non_vLine.end(), cmp);

    int n = 1;
    int nLength = non_vLine.size();
    std::vector<std::vector<non_vLine_s>> split_nv;
    int flag = 0;
    while (n < nLength) {
        std::vector<non_vLine_s> temp = { non_vLine[n - 1] };
        while (std::abs(non_vLine[n - 1].k - non_vLine[n].k) <= 5.) {
            temp.push_back(non_vLine[n]);
            n++;
            if (n == nLength) {
                flag = 1;
                split_nv.push_back(temp);
                break;
            }
        }
        if (n < nLength) {
            if (std::abs(non_vLine[n - 1].k - non_vLine[n].k) > 5.) {
                split_nv.push_back(temp);
            }
        }
        n++;
    }
    if (flag == 0) {
        std::vector<non_vLine_s> lastemp = { non_vLine.back() };
        split_nv.push_back(lastemp);
    }
    std::vector<double> confs;
    for (auto & sn : split_nv) {
        double conf = non_vLineConf(sn);
        confs.push_back(conf);
    }

    auto biggest = std::max_element(std::begin(confs), std::end(confs));
    double max_conf = *biggest;
    auto big_index = std::distance(std::begin(confs), biggest);

    nonvLineConf_value nv_result;
    nv_result.nv_x_conf = max_conf;
    nv_result.nv_main_line = split_nv[big_index];

    return nv_result;
}


static std::vector<std::vector<double>> filterLine(const std::vector<cv::Vec4i>& lines, const int ort) {


    std::vector<double> vLine_x;
    std::vector<vLine_s> vLine;
    std::vector<non_vLine_s> non_vLine;

    // output
    std::vector<std::vector<double>> v_m_lines(2);
    std::vector<std::vector<double>> nv_m_line_s(2);

    double x0, y0, x1, y1;
    for (const auto & line : lines) {
        x0 = double(line[0]);
        y0 = double(line[1]);
        x1 = double(line[2]);
        y1 = double(line[3]);
        if (x0 != x1) {
            double k = (y1 - y0) / (x1 - x0);
            double b = y0 - k * x0;
            non_vLine_s nvl_temp;
            nvl_temp.cdline[0] = x0;
            nvl_temp.cdline[1] = y0;
            nvl_temp.cdline[2] = x1;
            nvl_temp.cdline[3] = y1;
            nvl_temp.k = k;
            nvl_temp.b = b;
            nvl_temp.conf = confidence(x0, y0, x1, y1);
            non_vLine.push_back(nvl_temp);
        }
        else {
            double b = x0;
            vLine_s vl_temp;
            vl_temp.cdline[0] = x0;
            vl_temp.cdline[1] = y0;
            vl_temp.cdline[2] = x1;
            vl_temp.cdline[3] = y1;
            vl_temp.b = b;
            vl_temp.conf = confidence(x0, y0, x1, y1);

            if (vLine_x.empty()) {
                vLine_x.push_back(b);
                vLine.push_back(vl_temp);
            }
            else
            {
                size_t v = 0;
                for ( ; v < vLine.size(); v++) {
                    if (y0 < vLine[v].cdline[1]) {
                        vLine_x.insert(vLine_x.begin() + v, b);
                        vLine.insert(vLine.begin() + v, vl_temp);
                        break;
                    }
                }//c++ no for else
                if (v == vLine.size()) {
                    vLine_x.push_back(b);
                    vLine.push_back(vl_temp);
                }

            }
        }
    }


    double v_x_conf;
    //
    // judge vertical line confidence
    if (!vLine_x.empty()) {
        vLineConf_value v_result = vLineConf(vLine, vLine_x);
        v_x_conf = v_result.v_x_conf;
        std::vector<vLine_s> v_m_line = v_result.v_main_line;
        for (auto & m : v_m_line) {
            v_m_lines[0].push_back(m.cdline[0]);
            v_m_lines[0].push_back(m.cdline[2]);
            v_m_lines[1].push_back(m.cdline[1]);
            v_m_lines[1].push_back(m.cdline[3]);
        }
    }
    else {
        v_x_conf = 0.;
        fillUPdefault(v_m_lines, ort);
    }

    double nv_x_conf;
    if (!non_vLine.empty()) {
        nonvLineConf_value nv_result = nonVlConf(non_vLine);
        nv_x_conf = nv_result.nv_x_conf;
        std::vector<non_vLine_s> nv_m_line = nv_result.nv_main_line;
        for (auto & ml : nv_m_line) {
            nv_m_line_s[0].push_back(ml.cdline[0]);
            nv_m_line_s[0].push_back(ml.cdline[2]);
            nv_m_line_s[1].push_back(ml.cdline[1]);
            nv_m_line_s[1].push_back(ml.cdline[3]);
        }
    }

    else {
        nv_x_conf = 0.;
        fillUPdefault(nv_m_line_s, ort);
    }

    if (v_x_conf >= nv_x_conf) {
        return v_m_lines;
    }
    else {
        return nv_m_line_s;
    }

}


static bool judge(cv::Vec4i line_point, std::vector<double> limit) {

    auto p0x = double(line_point[0]);
    auto p0y = double(line_point[1]);
    auto p1x = double(line_point[2]);
    auto p1y = double(line_point[3]);

    double rx0 = limit[0];
    double ry0 = limit[1];
    double rx1 = limit[2];
    double ry1 = limit[3];

    if ((p1y - p0y) != 0.) {
        double sx = (ry0 - p0y) * (p1x - p0x) / (p1y - p0y) + p0x;
        if (rx0 <= sx && sx <= rx1) {
            return true;
        }
        double xx = (ry0 - p0y) * (p1x - p0x) / (p1y - p0y) + p0x;
        if (rx0 <= xx && xx <= rx1) {
            return true;
        }
    }
    else {
        if (ry0 < p0y && p0y < ry1) {
            return true;
        }
    }

    if ((p1x - p0x) != 0.) {
        double zy = (ry1 - p0y) * (rx1 - p0x) / (p1x - p0x) + p0y;
        if (ry0 <= zy && zy <= ry1) {
            return true;
        }
        double yy = (ry1 - p0y) * (rx1 - p0x) / (p1x - p0x) + p0y;
        if (ry0 >= yy && yy >= ry1) {
            return true;
        }
    }
    else {
        if (rx0 < p0x && p0x < rx1) {
            return true;
        }
    }

    return false;
}


static kandb linefit(std::vector<std::vector<double>>& lines, int mode) {

    std::vector<double> X = lines[0];
    std::vector<double> Y = lines[1];
    auto N = double(X.size());
    double sum_X = 0.;
    for (double i : X) {
        sum_X += i;
    }
    double sum_Y = 0.;
    for (double j : Y) {
        sum_Y += j;
    }
    double xavg = sum_X / N;
    double yavg = sum_Y / N;
    double k, b;
    if (mode == 0) {
        double sum_XY = 0.;
        double sum_XX = 0.;
        for (size_t i = 0; i < X.size(); i++) {
            sum_XY += X[i] * Y[i];
            sum_XX += X[i] * X[i];
        }
        double ly = sum_XY - N * xavg * yavg;
        double lt = sum_XX - N * xavg * xavg;
        k = ly / lt;
        b = yavg - k * xavg;
    }
    else {
        double sum_XY = 0.;
        double sum_YY = 0.;
        for (size_t i = 0; i < X.size(); i++) {
            sum_XY += X[i] * Y[i];
            sum_YY += Y[i] * Y[i];
        }
        double ly = sum_XY - N * xavg * yavg;
        double lt = sum_YY - N * yavg * yavg;
        k = ly / lt;
        b = xavg - k * yavg;
    }
    if (mode == 1) {
        try {
            double temp = k;
            if (temp != 0) {
                k = 1 / temp;
                b = -b / temp;
            }else {
                k = double(NAN);
                b = sum_X / N;
            }
            if (k>200.){
                k = double(NAN);
                b = sum_X / N;
            }
        }
        catch (...)
        {
            k = double(NAN);
            b = sum_X / N;
        }
    }
    kandb kwithb;
    kwithb.k = k;
    kwithb.b = b;

    return kwithb;
}


static std::vector<cv::Point2d> edgeLinePoint(double k, double b) {
    std::vector<cv::Point2d> fourEdges(4);
    double xl = 0.;
    double xr = 0.;
    if (isnan(k)) {
        fourEdges[0].x = b;
        fourEdges[0].y = double(0.);
        fourEdges[1].x = b;
        fourEdges[1].y = double(511.);
    }
    else if (k != 0.) {
        if (b != 0.) {
            xl = -b / k;
        }
        if ((511 - b) != 0.) {
            xr = (511 - b) / k;
        }
        fourEdges[0].x = xl;
        fourEdges[0].y = double(0.);
        fourEdges[1].x = xr;
        fourEdges[1].y = double(511.);
    }
    else {
        fourEdges[0].x = double(0.);
        fourEdges[0].y = b;
        fourEdges[1].x = double(511.);
        fourEdges[1].y = b;
    }

    double yu = b;
    double yd = 511 * k + b;
    fourEdges[2].x = double(0.);
    fourEdges[2].y = yu;
    fourEdges[3].x = double(511.);
    fourEdges[3].y = yd;

    std::vector<double> para = { xl, xr, yu, yd };

    std::vector<cv::Point2d> twoAxis;
    try {
        for (size_t p = 0; p < 4; p++) {
            if (para[p] >= 0. && para[p] < 512.) {
                twoAxis.push_back(fourEdges[p]);
                if (twoAxis.size() == 2) {
                    break;
                }
            }
        }
    }
    catch (...) {
        twoAxis[0].x = double(0.);
        twoAxis[0].y = double(0.);
        twoAxis[1].x = double(0.);
        twoAxis[1].y = double(0.);
    }

    return twoAxis;
}


static double det(const cv::Point2d& a, const cv::Point2d& b) {
    return a.x * b.y - a.y * b.x;
}


static cv::Point2d line_intersection(std::vector<cv::Point2d> line1, std::vector<cv::Point2d> line2) {


    cv::Point2d xdiff, ydiff, cross_point;

    xdiff.x = line1[0].x - line1[1].x;
    xdiff.y = line2[0].x - line2[1].x;
    ydiff.x = line1[0].y - line1[1].y;
    ydiff.y = line2[0].y - line2[1].y;

    double div = det(xdiff, ydiff);

    if (div == 0.) {
        return cross_point;
    }

    cv::Point2d d;
    d.x = det(line1[0], line1[1]);
    d.y = det(line2[0], line2[1]);
    cross_point.x = det(d, xdiff) / div;
    cross_point.y = det(d, ydiff) / div;

    return cross_point;
}


static std::vector<cv::Point2d> shortBox(kandb lkb, kandb tkb, kandb rkb, kandb dkb) {

    std::vector<cv::Point2d> result;
    std::vector<cv::Point2d> left_side, right_side, top_side, down_side;

    left_side = edgeLinePoint(lkb.k, lkb.b);
    right_side = edgeLinePoint(rkb.k, rkb.b);
    top_side = edgeLinePoint(tkb.k, tkb.b);
    down_side = edgeLinePoint(dkb.k, dkb.b);

    cv::Point2d lu = line_intersection(left_side, top_side);
    cv::Point2d ur = line_intersection(top_side, right_side);
    cv::Point2d rd = line_intersection(right_side, down_side);
    cv::Point2d ld = line_intersection(down_side, left_side);

    result = { lu, ur, rd, ld };

    return result;
}


static cv::Point2d cal_boundary_point(kandb coef, const cv::Point2d& ocs) {

    double x1 = 0.;
    double x2 = 511.;
    double y1 = 0.;
    double y2 = 511.;
    std::vector<cv::Point2d> cbp;
    if (isnan(coef.k)) {
        double X = coef.b;
        cv::Point2d temp;
        temp.x = X;
        temp.y = y1;
        cbp.push_back(temp);
        temp.x = X;
        temp.y = y2;
        cbp.push_back(temp);
    }
    else if (coef.k == 0.) {
        double Y = coef.b;
        cv::Point2d temp;
        temp.x = x1;
        temp.y = Y;
        cbp.push_back(temp);
        temp.x = x2;
        temp.y = Y;
        cbp.push_back(temp);
    }
    else {
        double Y1 = x1 * coef.k + coef.b;
        double Y2 = x2 * coef.k + coef.b;
        double X1 = (y1 - coef.b) / coef.k;
        double X2 = (y2 - coef.b) / coef.k;
        cv::Point2d temp;
        if (Y1 > y1 && Y1 < y2) {
            temp.x = x1;
            temp.y = Y1;
            cbp.push_back(temp);
        }
        if (Y2 > y1 && Y2 < y2) {
            temp.x = x2;
            temp.y = Y2;
            cbp.push_back(temp);
        }
        if (X1 > x1 && X1 < x2) {
            temp.x = X1;
            temp.y = y1;
            cbp.push_back(temp);
        }
        if (X2 > x1 && X2 < x2) {
            temp.x = X2;
            temp.y = y2;
            cbp.push_back(temp);
        }
    }

    std::vector<double> distance;
    for (auto & c : cbp) {
        double d = pow((c.x - ocs.x), 2) + pow((c.y - ocs.y), 2);
        distance.push_back(d);
    }

    auto biggest = std::max_element(std::begin(distance), std::end(distance));
    double max_dist = *biggest;
    auto big_index = std::distance(std::begin(distance), biggest);

    return cbp[big_index];

}


static void fixCorner(std::vector<kandb> &ori_kandb, std::vector<cv::Point2d> ori_points, int &flag) {

    int c = 0;
    while (c < 4) {
        cv::Point2d cc = ori_points[c];
        while (!((cc.x >= 0. && cc.x < 512.) && (cc.y >= 0. && cc.y < 512.))) {
            int post_c_idx = (c + 1) % 4;
            cv::Point2d pre_cbp = cal_boundary_point(ori_kandb[c], cc);
            cv::Point2d post_cbp = cal_boundary_point(ori_kandb[post_c_idx], cc);
            double d0 = pow(pre_cbp.x - cc.x, 2) + pow(pre_cbp.y - cc.y, 2);
            double d1 = pow(post_cbp.x - cc.x, 2) + pow(post_cbp.y - cc.y, 2);
            if (d0 <= d1) {
                if (isnan(ori_kandb[post_c_idx].k)) {
                    ori_kandb[post_c_idx].b = pre_cbp.x;
                }
                else {
                    ori_kandb[post_c_idx].b = pre_cbp.y - pre_cbp.x * ori_kandb[post_c_idx].k;
                }
            }
            else {
                if (isnan(ori_kandb[c].k)) {
                    ori_kandb[c].b = post_cbp.x;
                }
                else {
                    ori_kandb[c].b = post_cbp.y - post_cbp.x * ori_kandb[c].k;
                }
            }
            flag = 1;
            break;
        }
        c++;
    }
}


std::vector<cv::Point2d> ProcessEdgeImageV2(const cv::Mat &edge_image) {

    int picW = 512;
    int theW = 511.;
    assert(edge_image.rows == picW);
    assert(edge_image.cols == picW);

    int height = edge_image.rows;
    int width = edge_image.cols;

    cv::Mat gray_image(picW, picW, CV_8UC1);
    if (edge_image.channels() == 3) {
        cv::cvtColor(edge_image, gray_image, cv::COLOR_RGB2GRAY);
    }
    else if (edge_image.channels() == 4) {
        cv::cvtColor(edge_image, gray_image, cv::COLOR_RGBA2GRAY);
    }
    else {
        gray_image = edge_image.clone();
    }

    cv::Mat binary_image;
    cv::threshold(gray_image, binary_image, 128, 255, cv::THRESH_BINARY);

    std::vector<cv::Vec4i> linesP;
    HoughLinesP(binary_image, linesP, 1, CV_PI / 180, 10, 10, 10);


    std::vector<cv::Vec4i> left_x, right_x, top_x, down_x;

    std::vector<double> limit;
    limit.push_back(double(height * 0.4));
    limit.push_back(double(height * 0.4));
    limit.push_back(double(height * 0.6));
    limit.push_back(double(height * 0.6));

    std::vector<kandb> all_kandb;
    std::vector<cv::Point2d> all_corners;

    if (linesP.size() > 0) {
        double x1, y1, x2, y2;
        for (size_t i = 0; i < linesP.size(); i++) {
            x1 = double(linesP[i][0]);
            y1 = double(linesP[i][1]);
            x2 = double(linesP[i][2]);
            y2 = double(linesP[i][3]);

            if (judge(linesP[i], limit)) {
                continue;
            }

            if (std::abs(x1 - x2) < std::abs(y1 - y2)) {
                if ((x1 + x2) / 2 < limit[0]) {
                    left_x.push_back(linesP[i]);
                }
                else {
                    right_x.push_back(linesP[i]);
                }
            }
            else {
                if ((y1 + y2) / 2 < limit[1]) {
                    top_x.push_back(linesP[i]);
                }
                else {
                    down_x.push_back(linesP[i]);
                }
            }
        }

        //====================================================
        std::vector<std::vector<cv::Vec4i>> all_points;
        all_points.push_back(left_x);
        all_points.push_back(right_x);
        all_points.push_back(top_x);
        all_points.push_back(down_x);

        if (fillUP(all_points, theW)) {
            left_x = all_points[0];
            right_x = all_points[1];
            top_x = all_points[2];
            down_x = all_points[3];
        }
        //====================================================

        std::vector<std::vector<double>> left_edge = filterLine(left_x, 0); //"left"
        std::vector<std::vector<double>> right_edge = filterLine(right_x, 1);  //"right"
        std::vector<std::vector<double>> top_edge = filterLine(top_x, 2); //"top"
        std::vector<std::vector<double>> down_edge = filterLine(down_x, 3);  //"down"


        //====================================================

        //====================================================
        kandb left_kb = linefit(left_edge, 1);
        kandb right_kb = linefit(right_edge, 1);
        kandb top_kb = linefit(top_edge, 0);
        kandb down_kb = linefit(down_edge, 0);

        all_kandb = { left_kb,top_kb, right_kb, down_kb };
        all_corners = shortBox(left_kb, top_kb, right_kb, down_kb);
    }

    std::cout << "finish infer" << std::endl;

//    if (all_kandb.size() > 0) {
//        int flag = 0;
//        fixCorner(all_kandb, all_corners, flag);
//        while (flag) {
//            all_corners = shortBox(all_kandb[0], all_kandb[1], all_kandb[2], all_kandb[3]);
//            fixCorner(all_kandb, all_corners, flag);
//        }
//    }
//
    bool allZero = true;
    for(int i = 0;i < all_corners.size();i++){
        if(all_corners[i].x != 0 || all_corners[i].y != 0){
            allZero = false;
        }
    }

    if (all_kandb.size() == 0 || allZero) {
        all_corners.clear();
        cv::Point2d p_temp;
        p_temp.x = double(0.);
        p_temp.y = double(0.);
        all_corners.push_back(p_temp);
        p_temp.x = double(511.);
        p_temp.y = double(0.);
        all_corners.push_back(p_temp);
        p_temp.x = double(511.);
        p_temp.y = double(511.);
        all_corners.push_back(p_temp);
        p_temp.x = double(0.);
        p_temp.y = double(511.);
        all_corners.push_back(p_temp);
    }

    std::cout << "finish clear" << std::endl;
    return all_corners;
}