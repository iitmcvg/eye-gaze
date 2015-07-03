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

#include "constants.h"


using namespace cv;
using namespace dlib;
double vmax = 1.0 ;

image_window vel;
void display_velocity(double vx, double vy, double vz ) {

	cv::Mat speedometer = cv::imread("..res/blank_speedometer1.png", 1);
	
	double vel_abs = sqrt(vx*vx + vy*vy + vz*vz);
	double slope_angle = (-vel_abs*300.0)/vmax + 45.0;
	cv::Point center = cv::Point(speedometer.rows/2.0, speedometer.cols/2.0);

	double del_x = 100*cos(slope_angle*PI/180.0);
	double del_y = 100*sin(slope_angle*PI/180.0);

	cv::Point end_point = cv:: Point(speedometer.rows/2.0 + del_x , speedometer.cols/2.0 + del_y);

	cv::Point txt_pt = cv:: Point(speedometer.rows/2.0 -65,speedometer.rows/2.0 +75 );

	std::stringstream ss;
	ss << vel_abs;
	std::string text = ss.str();
	int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 0.75;
	int thickness = 2;
	cv::putText(speedometer, text, txt_pt, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);

	cv::circle( speedometer, center, 5,cv::Scalar( 0, 0, 0 ), -1, 8 );

	cv::line(speedometer, center, end_point, cv::Scalar (255 ,0 ,0), 3, 8, 0);

	vel.clear_overlay();
	vel.set_image(cv_image <bgr_pixel>(speedometer));


}