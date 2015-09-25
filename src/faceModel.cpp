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
#include "faceModel.h"

void preprocessROI(cv::Mat& roi_eye) {
	GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
	equalizeHist( roi_eye, roi_eye );
}

double find_sigma(int ln, int lf, double Rn, double theta) {
	double dz=0;
	double sigma;
	double m1 = ((double)ln*ln)/((double)lf*lf);
	double m2 = (cos(theta))*(cos(theta));

	if (m2 == 1)
	{
		dz = sqrt(	(Rn*Rn)/(m1 + (Rn*Rn))	);
	}
	if (m2>=0 && m2<1)
	{
		dz = sqrt(	((Rn*Rn) - m1 - 2*m2*(Rn*Rn) + sqrt(	((m1-(Rn*Rn))*(m1-(Rn*Rn))) + 4*m1*m2*(Rn*Rn)	))/ (2*(1-m2)*(Rn*Rn))	);
	}
	sigma = acos(dz);
	return sigma;
}

void faceModel::assign(full_object_detection shape , cv::Mat image, int mode = MODE_GAZE_VA) {
	assert(mode == MODE_GAZE_VA || mode == MODE_GAZE_QE);
	faceShape = shape;
	image.copyTo(inputImage);

	descriptors.clear();

	computePupil();
	computeNormal();
	computeGaze(mode);
}

void faceModel::computePupil() {
	// Computing left pupil
	std::vector<cv::Point> leftEyePoints = getFeatureDescriptors(INDEX_LEFT_EYE);
	rectLeftEye = cv::boundingRect(leftEyePoints)
	roiLeftEye = inputImage(rectLeftEye)
	preprocessROI(roiLeftEye);
	descriptors.push_back(get_pupil_coordinates(roiLeftEye,rectLeftEye));

	// Computing right pupil
	std::vector<cv::Point> rightEyePoints = getFeatureDescriptors(INDEX_RIGHT_EYE);
	rectRightEye = cv::boundingRect(rightEyePoints)
	roiRightEye = inputImage(rectRightEye)
	preprocessROI(roiRightEye);
	descriptors.push_back(get_pupil_coordinates(roiRightEye,rectRightEye));
}

void faceModel::computeNormal() {
	cv::Point midEye = get_mid_point(cv::Point(shape.part(39).x(), shape.part(39).y()),
		cv::Point(shape.part(40).x(), shape.part(40).y()));

	cv::Point mouth = get_mid_point(cv::Point(shape.part(48).x(), shape.part(48).y()),
		cv::Point(shape.part(54).x(), shape.part(54).y()));

	cv::Point noseTip = cv::Point(shape.part(30).x(), shape.part(30).y());
	cv::Point noseBase = cv::Point(shape.part(33).x(), shape.part(33).y());

	// symm angle - angle between the symmetry axis and the 'x' axis 
	symm_x = get_angle_between(noseBase, midEye);
	// tilt angle - angle between normal in image and 'x' axis
	tau = get_angle_between(noseBase, noseTip);
	// theta angle - angle between the symmetry axis and the image normal
	theta = (abs(tau - symm_x)) * (PI/180.0);

	// sigma - slant angle
	sigma = find_sigma(get_distance(noseTip, noseBase), get_distance(midEye, mouth), Rn, theta);

	normal[0] = (sin(sigma))*(cos((360 - tau)*(PI/180.0)));
	normal[1] = (sin(sigma))*(sin((360 - tau)*(PI/180.0)));
	normal[2] = -cos(sigma);

	pitch = acos(sqrt((normal[0]*normal[0] + normal[2]*normal[2])/(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])));
	if((noseTip.y - noseBase.y) < 0) {
		pitch = -pitch;
	}

	yaw = acos((abs(normal[2]))/(sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])));
	if((noseTip.x - noseBase.x) < 0) {
		yaw = -yaw;
	}
}

void computeGaze(int mode) {

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

std::vector<double> getNormal() {
	return normal;
}

cv::Point faceModel::getPupil(int mode) {
	assert(mode == INDEX_LEFT_EYE_PUPIL || mode == INDEX_RIGHT_EYE_PUPIL);
	return descriptors[mode - INDEX_LEFT_EYE_PUPIL];
}

std::vector<cv::Point> faceModel::getDescriptors(int index) {
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

	else if (index == INDEX_RIGHT_EYE_BROW || ) {

		std::vector<cv::Point> rightEyeBrowPoints;
		for (int i=22; i<=26; i++){
			rightEyeBrowPoints.push_back(cv::Point(faceShape.part(i).x(), faceShape.part(i).y()));
		}
		return rightEyeBrowPoints;
	}

	else if (index == INDEX_NOSE_UPPER || )  {

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