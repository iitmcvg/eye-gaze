#include <math.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

void filter_image(cv::Mat roi ) {

	std::vector<double> cdf(256);
	double nf, temp, pos_i, pos_j;

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
		std::cout<<"CDF-"<<i<<" = "<<cdf[i]<<std::endl;
	}

	nf = cdf[0];
	for(int i=1; i<256;i++) {
		if(cdf[i] > nf) {
			nf = cdf[i];
		}
	}

	temp = roi.at<uchar>(0,0);
	pos_i = 0;
	pos_j = 0;

	for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			if(cdf[roi.at<uchar>(i,j)] >= 0.05 * nf) {
				//roi.at<uchar>(i,j) = 255;
			}
			else {
				if(roi.at<uchar>(i,j) <= temp) {
					pos_i = i;
					pos_j = j;
					temp = roi.at<uchar>(i,j);
				}
				//roi.at<uchar>(i,j) = 255;
			}
		}
	}
	roi.at<uchar>(pos_i,pos_j) = 255;
}
