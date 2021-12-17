#include<opencv2/opencv.hpp>
#include<iostream>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;


double getDistance(Point2f point1, Point2f point2)
{
	double distance = sqrtf(powf((point1.x - point2.x), 2) + powf((point1.y - point2.y), 2));

	return distance;
}


Mat correct(Mat& raw_img,vector<Point2f> &corners)
{
	Mat gray;
	if (raw_img.channels() == 3)
	{

		cvtColor(raw_img, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = raw_img;
	}
	float h_rows = raw_img.rows;
	float w_cols = raw_img.cols;
	float min_x = w_cols, min_y = h_rows, max_x = 0, max_y = 0;


	for (int i = 0; i < corners.size(); ++i)
	{
		min_x = (corners[i].x > min_x) ? min_x : corners[i].x;
		min_y = (corners[i].y > min_y) ? min_x : corners[i].y;
		max_x = (corners[i].x < max_x) ? max_x : corners[i].x;
		max_y = (corners[i].y < max_y) ? max_y : corners[i].y;
	}
	Point2d lt(0, 0);
	int temp_x = int(max_x - min_x);
	int temp_y = int(max_y - min_y);
	Point2d rt(temp_x, 0);
	Point2d rd(temp_x, temp_y);
	Point2d ld(0, temp_y);
	vector<Point2d> aimp;

	aimp.push_back(lt);
	aimp.push_back(rt);
	aimp.push_back(rd);
	aimp.push_back(ld);

	vector<Point2f> cc(4);
	cc.swap(corners);

	vector<Point2f> pts1;
	vector<Point2f> pts2;
	vector<Point2f> new_cornes;
	for (auto p : aimp)
	{
		pts2.push_back(p);
		vector<float> td;
		for (auto c : cc)
		{
			Point2d dd(p.x + min_x, p.y + min_y);
			float temp = getDistance(dd, c);
			td.push_back(temp);
		}
		auto cid = min_element(td.begin(), td.end());
		int index = distance(begin(td), cid);
		pts1.push_back((cc[index]));
	}
	Mat result_images;
	Mat warpmatrix = getPerspectiveTransform(pts1, pts2);
	warpPerspective(gray, result_images, warpmatrix, cv::Size(int(max_x - min_x),int(max_y - min_y))); //͸�ӱ任
	//cv::imshow("src", result_images);

	//cv::waitKey();
	return result_images;
}