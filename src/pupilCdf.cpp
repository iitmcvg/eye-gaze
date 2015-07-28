#include <math.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

void filter_image(cv::Mat roi ) {

	int histSize[] = {256};
	float range[] = { 0, 255 } ; 
	const float* histRange[] = { range };
	bool uniform = true;
	bool accumulate = false;
	int* channels = {0};
	cv::Mat histogram;
    cv::Mat temproi;
    roi.copyTo(temproi);

	cv::calcHist( &temproi, 1, channels, cv::Mat(), histogram, 1, histSize, histRange, uniform, accumulate );

	std::vector<double> cdf(256);

	//cv::normalize(histogram, histogram, 0, roi.rows*roi.cols, NORM_MINMAX, -1, Mat() );
	double sum = 0;
	for(int i=0; i < histSize[0]; i++) {
		double value;

		if(i == 0) {
			value = (histogram.at<float>(i));
        }
		else {
			value = (histogram.at<float>(i)) + cdf[i-1];
		}
		
		std::cout<<value<<std::endl;

		cdf[i] = value;
		sum += value;
		//std::cout<<(int)roi.at<uchar>(10,10)<<std::endl;
		std::cout<<"size "<<histogram.size()<<" "<<histogram.at<float>(i)<<std::endl;
	}

	/*for(int i = 0; i < roi.rows; i++) {
		for(int j = 0; j < roi.cols; j++) {
			if(cdf[roi.at<uchar>(j,i)] < 0.05 * sum) {
				roi.at<uchar>(j,i) = 0;
			}
			else {
				roi.at<uchar>(j,i) = 255;
			}
		}
	}*/
    return;
}
