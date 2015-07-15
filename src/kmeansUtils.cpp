#include <math.h>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

using namespace std;


void kmeans_array_generate(cv::Mat src, std::vector<float>& vec, int mode) {
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

	for(int i=0; i<src.cols*src.rows; i++) {

		vec.push_back(hsv[0].data[i] / 255.0) ;
	}
}

void kmeans_clusters_view(cv::Mat& src, std::vector<int> vec_labels) {
	int rows = src.rows;
	int cols = src.cols;
	int clr;
	int idx = 0;


	for(int i=0; i<src.cols*src.rows; i++) {

		if(((int)(vec_labels[i])) == 2) {
			clr = 255;
		}
		else {
			clr = 0;
		}

		src.at<float>(i/src.cols, i%src.cols) = clr;

	}


}
