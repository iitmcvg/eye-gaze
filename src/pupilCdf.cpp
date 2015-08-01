#include <math.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

void filter_image(cv::Mat roi ) {

	std::vector<double> cdf(256);
	double temp;

	for(int i = 0; i<256; i++) {
		cdf[i] = 0;
	}

	for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			++cdf[roi.at<uchar>(i,j)];
		}
	}

	for(int i=0; i < 256; i++) {
		double value;
		if(i != 0) {
			value = cdf[i];
			cdf[i] += cdf[i-1];
		}
		std::cout<<"CDF- "<<i<<" = "<<cdf[i]<<std::endl;
	}

	temp = cdf[0];
	for(int i=1; i<256;i++) {
		if(cdf[i] > temp) {
			temp = cdf[i];
		}
	}

	std::cout<<"Thresh : "<<0.05 * roi.rows * roi.cols<<std::endl;

	for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			if(cdf[roi.at<uchar>(i,j)] >= 0.05 * temp) {
				roi.at<uchar>(i,j) = 0;
			}
			else {
				roi.at<uchar>(i,j) = 255;
			}
		}
	}
}
