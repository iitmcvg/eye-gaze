#include <math.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

void filter_image(cv::Mat roi_clr, cv::Point& pt_pupil) {

	std::vector<double> cdf(256);
	cv::Mat roi;
	cv::cvtColor(roi_clr, roi, CV_BGR2GRAY);

	//Preprocessing
    GaussianBlur(roi, roi, cv::Size(3,3), 0, 0);
	cv::equalizeHist(roi, roi);

	cv::Mat mask;
	roi.copyTo(mask);
	double nf, temp, pos_pmi_i, pos_pmi_j;

	int erosion_size = 1;
	cv::Mat element_erode = cv::getStructuringElement( cv::MORPH_ELLIPSE,
		cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		cv::Point( erosion_size, erosion_size ) );

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
		//std::cout<<"CDF-"<<i<<" = "<<cdf[i]<<std::endl;
	}

	nf = cdf[0];
	for(int i=1; i<256;i++) {
		if(cdf[i] > nf) {
			nf = cdf[i];
		}
	}

	temp = roi.at<uchar>(0,0);
	pos_pmi_i = 0;
	pos_pmi_j = 0;

	for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			if(cdf[roi.at<uchar>(i,j)] >= 0.05 * nf) {
				mask.at<uchar>(i,j) = 0;
				//roi.at<uchar>(i,j) = 255;
			}
			else {
				if(roi.at<uchar>(i,j) <= temp) {
					pos_pmi_i = i;
					pos_pmi_j = j;
					temp = roi.at<uchar>(i,j);
				}
				mask.at<uchar>(i,j) = 255;
				//roi.at<uchar>(i,j) = 255;
			}
		}
	}

	cv::erode( mask, mask, element_erode );

	double avg_PI = 0;

	for(int i = pos_pmi_i - 5; i < pos_pmi_i + 5; i++) {
		for(int j = pos_pmi_j - 5; j < pos_pmi_j + 5; j++) {
			if(mask.at<uchar>(i,j)) {
				avg_PI += roi.at<uchar>(i,j);
			}
		}
	}

	for(int i = pos_pmi_i - 5; i < pos_pmi_i + 5; i++) {
		for(int j = pos_pmi_j - 5; j < pos_pmi_j + 5; j++) {
			if(roi.at<uchar>(i,j) > ((int)avg_PI)) {
				mask.at<uchar>(i,j) = 0;
			}
		}
	}

	cv::erode( mask, mask, element_erode );

	//mask.copyTo(roi);

	cv::Moments m = cv::moments(mask, 1);
	int pos_i = (int)(m.m10/m.m00), pos_j = (int)(m.m01/m.m00);

	std::cout<<"PMI : "<<pos_pmi_i<<", "<<pos_pmi_j<<std::endl;
	std::cout<<"Point : "<<pos_i<<", "<<pos_j<<std::endl;

	pt_pupil.x = pos_i;
	pt_pupil.y = pos_j;

	cv::circle(roi_clr, cv::Point(pos_i, pos_j), 3, cv::Scalar(255,0,0), -1, 4, 0);
}