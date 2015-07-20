#ifndef FACE_ANALYSIS_H
#define FACE_ANALYSIS_H

struct FaceFeatures {
	cv::Point face_centre;

	cv::Point left_eye;
	cv::Point right_eye;
	cv::Point mid_eye;

	cv::Point nose_base;
	cv::Point nose_tip;

	cv::Point mouth;

	void assign(cv::Point c_face_centre, cv::Point c_left_eye, cv::Point c_right_eye, cv::Point c_nose_tip, cv::Point c_mouth);
};

struct FaceData {
	double left_eye_nose_distance;
	double right_eye_nose_distance;
	double left_eye_right_eye_distance;
	double nose_mouth_distance;

	double mid_eye_mouth_distance;	//Lf
	double nose_base_nose_tip_distance;	//Ln

	void assign(FaceFeatures* f);
};

struct FacePose {
	double theta, tau;
	double sigma, symm_x;

	double normal[3];	//Vector for storing Facial normal

	double yaw, pitch;

	double kalman_pitch, kalman_yaw;
	double kalman_pitch_pre, kalman_yaw_pre;

	void assign(FaceFeatures* f, FaceData* d);
};

void draw_facial_normal(cv::Mat& img, dlib::full_object_detection shape, std::vector<double> normal);
void draw_crosshair(cv::Mat img, CvPoint centre, int circle_radius, int line_radius);
void project_facial_pose(cv::Mat img, double normal[3], double sigma, double theta);
double find_sigma(int ln, int lf, double Rn, double theta);

#endif