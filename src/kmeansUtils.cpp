#include <math.h>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

using namespace std;

void kmeans_array_generate(cv::Mat src, std::vector<std::vector<double> >& vec, int mode) {
	if(vec.size() != 0) {
		vec.clear();
	}

	int rows = src.rows;
	int cols = src.cols;

	vec.resize(cols*rows);

	int idx = 0;

	cv::Mat src_hsv;
	cv::cvtColor(src, src_hsv, CV_RGB2HSV);

	std::vector<cv::Mat> hsv;
	cv::split(src, hsv);

	for(int i=0;i<rows;i++) {
		for(int j=0;j<cols;j++) {
			//vec[idx].push_back(((double) j)/cols);
			//vec[idx].push_back(((double) i)/rows);
			//std::cout<<"hue : "<<((double)(hsv[0].at<uchar>(j,i)))<<std::endl;
			vec[idx].push_back(((double) hsv[0].at<uchar>(j, i)));

			idx++;
		}
	}
}

void kmeans_clusters_view(cv::Mat& src, cv::Mat labels) {
	int rows = src.rows;
	int cols = src.cols;

	int idx = 0;
	std::cout<<labels.size()<<"\t"<<rows;
	for(int i=0;i<rows;i++) {
		for(int j=0;j<cols;j++) {
			int clr = labels.at<uchar>(0, idx)*100 + 50;

			src.at<cv::Vec3b>(i, j)[0] = clr;
			src.at<cv::Vec3b>(i, j)[1] = clr;
			src.at<cv::Vec3b>(i, j)[2] = clr;

			std::cout<<"val0 "<<src.at<cv::Vec3b>(i, j)[0]<<"\t";

			idx++;
		}
	}
}