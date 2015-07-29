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

void compute_eye_gaze (FacePose* face_pose, dlib::full_object_detection shape, cv::Point pupil, double mag_CP, double mag_LR) {

	std::vector<double> vec_LR_u(3), vec_RP(3), vec_CR_u(3), vec_CM_u(3), vec_UD_u(3), vec_CP(3);

	cv::Point p1, p2;
    //mode : 0 for left eye, 1 for right eye
	if(mode == 0) {
		p1 = cv::Point(shape.part(42).x(), shape.part(42).y());
		p2 = cv::Point(shape.part(45).x(), shape.part(45).y());
	}
	else if(mode == 1) {
		p1 = cv::Point(shape.part(36).x(), shape.part(36).y());
		p2 = cv::Point(shape.part(39).x(), shape.part(39).y());
	}

	compute_vec_LR(p1, p2, vec_LR_u);
	make_unit_vector(vec_LR_u, vec_LR_u);

	vec_CM_u[0] = face_pose->normal[0];
	vec_CM_u[1] = face_pose->normal[1];
	vec_CM_u[2] = face_pose->normal[2];

	cross_product(vec_CM_u, vec_LR_u, vec_UD_u);
	make_unit_vector(vec_UD_u, vec_UD_u);

	solve_CR(vec_UD_u, vec_CM_u, vec_CR_u);
	make_unit_vector(vec_CR_u, vec_CR_u);

	get_section(p1, p2, pupil, Y1, Y2);
	//Vector RP is in real world coordinates
	compute_vec_CP(vec_LR_u, mag_LR, vec_CP, Y1, Y2);



}

void compute_vec_LR (cv::Point p1, cv::Point p2, FacePose* face_pose, std::vector<double>& LR) {
	LR[0] = p1.x - p2.x;
	LR[1] = p1.y - p2.y;
	LR[2] = -(LR[0]*face_pose->normal[0] + LR[1]*face_pose->normal[1])/face_pose->normal[3];
}

void cross_product(std::vector<double> vec1, std::vector<double> vec2, std::vector<double> product) {
	product[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
	product[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
	product[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
}

void solve(std::vector<double> coeff_1, double const_1, std::vector<double> coeff_2, double const_2, double mag, std::vector<double>& vec) {

}

void get_section(cv::Point p1, cv::Point p2, cv::Point pupil, double& Y1, double& Y2, double& h) {
	std::vector<double> line(3);
	line[0] = p2.y - p1.y;
	line[1] = -(p2.x - p1.x);
	line[2] =  p1.y*(p2.x - p1.x) - p1.x*(p2.y - p1.y);

	cv::Point pupil_proj;
	pupil_proj.x = -(line[0]*pupil.x + line[1]*pupil.y + line[2])*line[0]/(line[0]*line[0] + line[1]*line[1]) + pupil.x;
	pupil_proj.y = -(line[0]*pupil.x + line[1]*pupil.y + line[2])*line[1]/(line[0]*line[0] + line[1]*line[1]) + pupil.y;

	Y1 = get_distance (p1, pupil_proj);
	Y2 = get_distance (p2, pupil_proj);
	h = get_distance (pupil, pupil_proj);

}

void compute_vec_CP(cv::Point p1, cv::Point p2, cv::Point pupil, FacePose* face_pose, std::vector<double> vec_CR_u, double mag_CR, std::vector<double> vec_LR_u, double mag_LR, std::vector<double> vec_UD_u, std::vector<double> vec_CP) {
	double Y1, Y2, H;
	get_section(p1, p2, pupil, Y1, Y2, H);

	double S2R;

	

}