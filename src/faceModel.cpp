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
#include "pupilDetection.h"

// using namespace std;
// using namespace dlib;

void faceModel::assign(full_object_detection shape , cv::Mat image) {
	faceShape = shape;
	image.copyTo(inputImage);
}

cv::Point faceModel::getPupil(int mode) {

	assert(mode == MODE_LEFT || mode == MODE_RIGHT);

	void preprocessROI(cv::Mat& roi_eye) {
		GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
		equalizeHist( roi_eye, roi_eye );
	}
	
	if (mode == MODE_LEFT) {
		std::vector<cv::Point> leftEyePoints = getFeatureDescriptors(INDEX_LEFT_EYE);
		rectLeftEye = cv::boundingRect(leftEyePoints)
		roiLeftEye = inputImage(rectLeftEye)
		preprocessROI(roiLeftEye);
		cv::Point pupilLeft = get_pupil_coordinates(roiLeftEye,rectLeftEye);
		return pupilLeft;
	}
	if (mode == MODE_RIGHT) {
		std::vector<cv::Point> rightEyePoints = getFeatureDescriptors(INDEX_RIGHT_EYE);
		rectRightEye = cv::boundingRect(rightEyePoints)
		roiRightEye = inputImage(rectRightEye)
		preprocessROI(roiRightEye);
		cv::Point pupilRight = get_pupil_coordinates(roiRightEye,rectRightEye);
		return pupilRight;
	}
}


std::vector<cv::Point> faceModel::getFeatureDescriptors(int index) {

	assert(index == INDEX_LEFT_EYE || index == INDEX_RIGHT_EYE || index == INDEX_LEFT_EYE_BROW || index == INDEX_RIGHT_EYE_BROW 
		|| index == INDEX_NOSE_UPPER || index == INDEX_NOSE_LOWER || index == INDEX_MOUTH_OUTER || index == INDEX_MOUTH_INNER); 

	if (index == INDEX_LEFT_EYE) {

		std::vector<cv::Point> leftEyePoints;
		for (int i=36; i<=41; i++){
			leftEyePoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return leftEyePoints;
	}

	else if (index == INDEX_RIGHT_EYE) {

		std::vector<cv::Point> rightEyePoints;
		for (int i=42; i<=47; i++){
			rightEyePoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return rightEyePoints;
	}

	else if (index == INDEX_LEFT_EYE_BROW) {

		std::vector<cv::Point> leftEyeBrowPoints;
		for (int i=17; i<=21; i++){
			leftEyeBrowPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return leftEyeBrowPoints;
	}

	else if (index == INDEX_RIGHT_EYE_BROW) {

		std::vector<cv::Point> rightEyeBrowPoints;
		for (int i=22; i<=26; i++){
			rightEyeBrowPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return rightEyeBrowPoints;
	}

	else if (index == INDEX_NOSE_UPPER)  {

		std::vector<cv::Point> NoseUpperPoints;
		for (int i=27; i<=30; i++){
			NoseUpperPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return NoseUpperPoints;
	}

	else if (index == INDEX_NOSE_LOWER) {
		
		std::vector<cv::Point> NoseLowerPoints;
		for (int i=31; i<=35; i++){
			NoseLowerPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return NoseLowerPoints;
	}

	else if (index == INDEX_MOUTH_OUTER) {

		std::vector<cv::Point> MouthOuterPoints;
		for (int i=48; i<59; i++){
			MouthOuterPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return MouthOuterPoints;
	}

	else if (index == INDEX_MOUTH_INNER) {

		std::vector<cv::Point> MouthInnerPoints;
		for (int i=60; i<=67; i++){
			MouthInnerPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return MouthInnerPoints;
	}
}

void faceModel::setOrigin(cv::Point origin) {
	this.origin = origin;
}

void faceModel::setOrigin(int mode) {
	assert(mode == ORIGIN_IMAGE || mode == ORIGIN_FACE_CENTRE);

	if (mode == ORIGIN_IMAGE) {
		origin.x = 0;
		origin.y = 0;
	}
	else if (mode == ORIGIN_FACE_CENTRE) {
		origin.x = shape.part(30).x();
		origin.y = shape.part(30).y();
	}
}