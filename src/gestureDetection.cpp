#include <math.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

#include "util.h"
#include "gestureDetection.h"

#define DTW_INFINITY 1e30

double maximum(double a, double b, double c) {
	if(a>b && a>c) {
		return a;
	}
	else if(b>a && b>c) {
		return b;
	}
	else return c;
}

double measure_deviation(std::vector<double> arr1, std::vector<double> arr2) {

	/*
		Function to find deviation/difference between two measured vectors having size = 3.

		@params:
		arr1	Input array 1
		arr2	Input array 2
	*/

	return maximum(std::fabs(arr1[0] - arr2[0]), std::fabs(arr1[1] - arr2[1]), std::fabs(arr1[2] - arr2[2]));
}

double minimum(double a, double b, double c) {
	if(a<b && a<c) {
		return a;
	}
	else if(b<a && b<c) {
		return b;
	}
	else return c;
}

double DTWScore(std::vector<std::vector<double> > arr1, std::vector<std::vector<double> > arr2) {

	/*
		Function to estimate how close two measurements are. The closeness is denoted by the score, which is computed
		using the DTW(Dynamic Time Warping) Algorithm. Lower the score, more close they are.

		@params:
		arr1	Input array 1
		arr2	Input array 2
	*/

	int m = arr1.size() - 1;
	int n = arr2.size() - 1;

	double DTW[m+1][n+1], dev;

	for(int i=1; i<=m; i++) {
		DTW[i][0] = DTW_INFINITY;
	}
	for(int j=1; j<=n; j++) {
		DTW[0][j] = DTW_INFINITY;
	}
	DTW[0][0] = 0;

	for(int i=1; i<=m; i++) {
		for(int j=1; j<=n; j++) {
			dev = measure_deviation(arr1[i], arr2[j]);
			DTW[i][j] = dev + minimum(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1]);
		}
	}

	//DTW[m][n] is the score.
	return DTW[m][n];
}

void FixedBin::assign(int _size) {
	filled = 0;
	size = _size;
	bin.resize(_size);
}

void FixedBin::push(std::vector<double> vec) {
	if(filled == size) {
		for(int i=0;i<filled-1;i++) {
			bin[i] = bin[i+1];
		}
		bin[filled - 1] = vec;
	}
	else {
		bin.at(filled) = vec;
		++filled;
	}
}

int FixedBin::get_size() {
	return size;
}

int FixedBin::get_filled() {
	return filled;
}

void FixedBin::get(int pos, std::vector<double>& vec) {
	vec.resize(3);
	vec = bin.at(pos);
}

std::vector<std::vector<double> > FixedBin::clone() {
	std::vector<std::vector<double> > vec(filled);

	//vec.empty();
	//vec.resize(filled);
	for(int i=0;i<filled;i++) {
		vec.at(i) = bin.at(i);
	}
	return vec;
}

void FaceGesture::assign(int normal_size) {
	normal = new FixedBin();
	normal->assign(normal_size);
}