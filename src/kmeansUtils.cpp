#include <math.h>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

using namespace std;

void kmeans_array_generate(cv::Mat src, std::vector<std::vector<double> > vec, int mode) {
	if(vec.size() != 0) {
		vec.clear();
	}

	int rows = src.rows;
	int cols = src.cols;

	int idx = 0;

	cv::Mat src_hsv;
	cv::cvtColor(src, src_hsv, CV_BGR2HSV);

	std::vector<cv::Mat> hsv;
	cv::split(src, hsv);

	for(int i=0;i<rows;i++) {
		for(int j=0;j<cols;j++) {
			vec[idx][0] = ((double) j)/cols;
			vec[idx][1] = ((double) i)/rows;
			vec[idx][2] = ((double) hsv[0].at<uchar>(i, j));

			idx++;
		}
	}
}

void kmeans_clusters_view(cv::Mat& src, std::vector<std::vector<double> > vec_labels) {
	int rows = src.rows;
	int cols = src.cols;

	int idx = 0;

	for(int i=0;i<rows;i++) {
		for(int j=0;j<cols;j++) {
			int clr = vec_labels[idx]*100 + 50;
			src.at<Vec3b>(i, j)[0] = clr;
			src.at<Vec3b>(i, j)[1] = clr;
			src.at<Vec3b>(i, j)[2] = clr;
		}
	}
}