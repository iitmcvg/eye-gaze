#include <math.h>
#include <stdlib.h>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

#include "util.h"
#include "constants.h"
#include "faceDetection.h"

// using namespace std;
// using namespace dlib;

void faceModel::assign(full_object_detection shape) {
	faceShape = shape;
}

cv::Point faceModel::getPupil(int mode) {


}

std::vector<cv::Point> faceModel::getFeatureDescriptors(int index) {

	if (index == INDEX_LEFT_EYE) {

		cv::Point points[] = {cv::Point(faceShape.part(36).x(), faceShape.part(36).y()), cv::Point(faceShape.part(37).x(), faceShape.part(37).y()), 
			cv::Point(faceShape.part(38).x(), faceShape.part(38).y()), cv::Point(faceShape.part(39).x(), faceShape.part(39).y()), 
			cv::Point(faceShape.part(40).x(), faceShape.part(40).y()), cv::Point(faceShape.part(41).x(), faceShape.part(41).y())};
		
		std::vector<cv::Point> leftEyePoints;
		for (int i=0; i<6; i++){
			leftEyePoints.push_back(points[i]);
		}
	return leftEyePoints;
	}

	else if (index == INDEX_RIGHT_EYE) {

		cv::Point points[] = {cv::Point(faceShape.part(42).x(), faceShape.part(42).y()), cv::Point(faceShape.part(43).x(), faceShape.part(43).y()), 
			cv::Point(faceShape.part(44).x(), faceShape.part(44).y()), cv::Point(faceShape.part(45).x(), faceShape.part(45).y()), 
			cv::Point(faceShape.part(46).x(), faceShape.part(46).y()), cv::Point(faceShape.part(47).x(), faceShape.part(47).y())};
		
		std::vector<cv::Point> rightEyePoints;
		for (int i=0; i<6; i++){
			rightEyePoints.push_back(points[i]);
		}
	return rightEyePoints;
	}

	else if (index == INDEX_LEFT_EYE_BROW) {

		cv::Point points[] = {cv::Point(faceShape.part(17).x(), faceShape.part(17).y()), cv::Point(faceShape.part(18).x(), faceShape.part(18).y()), 
			cv::Point(faceShape.part(19).x(), faceShape.part(19).y()), cv::Point(faceShape.part(20).x(), faceShape.part(20).y()), 
			cv::Point(faceShape.part(21).x(), faceShape.part(21).y())};
		
		std::vector<cv::Point> leftEyeBrowPoints;
		for (int i=0; i<5; i++){
			leftEyeBrowPoints.push_back(points[i]);
		}
	return leftEyeBrowPoints;
	}

	else if (index == INDEX_RIGHT_EYE_BROW) {

		cv::Point points[] = {cv::Point(faceShape.part(22).x(), faceShape.part(22).y()), cv::Point(faceShape.part(23).x(), faceShape.part(23).y()), 
			cv::Point(faceShape.part(24).x(), faceShape.part(24).y()), cv::Point(faceShape.part(25).x(), faceShape.part(25).y()), 
			cv::Point(faceShape.part(26).x(), faceShape.part(26).y())};
		
		std::vector<cv::Point> rightEyeBrowPoints;
		for (int i=0; i<5; i++){
			rightEyeBrowPoints.push_back(points[i]);
		}
	return rightEyeBrowPoints;
	}
}
