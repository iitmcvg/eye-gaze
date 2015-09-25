#ifndef FACE_MODEL_H
#define FACE_MODEL_H

class Face {

private:
	static const int INDEX_LEFT_EYE = 0;
	static const int INDEX_LEFT_EYEBROW = 1;
	static const int INDEX_RIGHT_EYE = 2;
	static const int INDEX_RIGHT_EYEBROW = 3;
	static const int INDEX_NOSE_UPPER = 4;
	static const int INDEX_NOSE_LOWER = 5;	
	static const int INDEX_MOUTH_OUTER = 6;
	static const int INDEX_MOUTH_INNER = 7;
	static const int INDEX_LEFT_EYE_PUPIL = 8;
	static const int INDEX_RIGHT_EYE_PUPIL = 9;

	static const int ORIGIN_IMAGE = 0;
	static const int ORIGIN_FACE_CENTRE = 1;

	full_object_detection faceShape;
	cv::Point origin = cv::Point(0,0);

	double yaw, pitch, sigma, symm_x, theta, tau;
	vector<double> normal, gaze;

	vector<cv::Point> descriptors;
	/*
		0 : INDEX_LEFT_EYE_PUPIL
		1 : INDEX_RIGHT_EYE_PUPIL
	*/

public:
    void assign(full_object_detection shape);
    cv::Point getPupil(int mode);
    std::vector<cv::Point> getDescriptors(int index);
    std::vector<double> getGaze();
    std::vector<double> getNormal();
    void setOrigin(cv::Point origin);
    void setOrigin(int mode);

};

#endif