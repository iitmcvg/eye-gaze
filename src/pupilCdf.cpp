#include <math.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

void filter_image(cv::Mat& roi ) {

	int histSize = 256;
	float range[] = { 0, 256 } ; 
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat histogram;


	cv::calcHist( &roi, 1, 0, roi, histogram, 1, &histSize, &histRange, uniform, accumulate );

	std::vector<double> cdf(256);

	//cv::normalize(histogram, histogram, 0, roi.rows*roi.cols, NORM_MINMAX, -1, Mat() );

	for(int i=0; i < histSize; i++) {
		/*double value;

		if(i = 0) {
			value = (histogram.at<float>(i));
		}
		else {
			value = (histogram.at<float>(i)) + cdf[i-1];
		}
		
		std::cout<<value<<std::endl;

		cdf[i] = value;*/
	std::cout<<(int)roi.at<uchar>(10,10)<<std::endl;
		std::cout<<"size "<<histogram.size()<<" "<<histogram.at<float>(i)<<std::endl;
	}

	for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			if(cdf[roi.at<uchar>(j,i)] < 0.05) {
				roi.at<uchar>(j,i) = 0;
			}
			else {
				roi.at<uchar>(j,i) = 255;
			}
		}
	}
}