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

using namespace std;
using namespace dlib;

cv::RNG rng = cv::RNG(-1);

cv::KalmanFilter KF_p_l (6,6,0);
cv::Mat_<float> measurement_p_l (6,1);

void init_kalman_point_p_l(cv::Point pt_pos_l) {
	KF_p_l.statePre.at<float>(0) = pt_pos_l.x;
	KF_p_l.statePre.at<float>(1) = pt_pos_l.y;
	KF_p_l.statePre.at<float>(2) = 0;
	KF_p_l.statePre.at<float>(3) = 0;
	KF_p_l.statePre.at<float>(4) = 0;
	KF_p_l.statePre.at<float>(5) = 0;

	/*KF_p_l.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_p_l.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);*/
	KF_p_l.transitionMatrix = *(cv::Mat_<float>(6,6) << 1,0,1,0,0.5,0,    0,1,0,1,0,0.5,     0,0,1,0,1,0,   0,0,0,1,0,1,  0,0,0,0,1,0,  0,0,0,0,0,1);
	rng.fill(KF_p_l.processNoiseCov, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(1));
	/*KF_p_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);*/

	cv::setIdentity(KF_p_l.measurementMatrix);
	cv::setIdentity(KF_p_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_p_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_p_l.errorCovPost, cv::Scalar::all(.1)); 
}

cv::Point2f kalman_correct_point_p_l(cv::Point pt_pos_l, cv::Point pt_pos_l_old, cv::Point pt_vel_old) {
	cv::Mat prediction = KF_p_l.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_p_l(0) = pt_pos_l.x;
	measurement_p_l(1) = pt_pos_l.y;
	measurement_p_l(2) = pt_pos_l.x - pt_pos_l_old.x;
	measurement_p_l(3) = pt_pos_l.y - pt_pos_l_old.y;
	measurement_p_l(4) = measurement_p_l(2) - pt_vel_old.x;
	measurement_p_l(5) = measurement_p_l(3) - pt_vel_old.y;

	cv::Mat estimated = KF_p_l.correct(measurement_p_l);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}

cv::KalmanFilter KF_p_r (6,6,0);
cv::Mat_<float> measurement_p_r (6,1);

void init_kalman_point_p_r(cv::Point pt_pos_r) {
	KF_p_r.statePre.at<float>(0) = pt_pos_r.x;
	KF_p_r.statePre.at<float>(1) = pt_pos_r.y;
	KF_p_r.statePre.at<float>(2) = 0;
	KF_p_r.statePre.at<float>(3) = 0;
	KF_p_r.statePre.at<float>(4) = 0;
	KF_p_r.statePre.at<float>(5) = 0;

	/*KF_p_r.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_p_r.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);*/
	KF_p_r.transitionMatrix = *(cv::Mat_<float>(6,6) << 1,0,1,0,0.5,0,    0,1,0,1,0,0.5,     0,0,1,0,1,0,   0,0,0,1,0,1,  0,0,0,0,1,0,  0,0,0,0,0,1);
	rng.fill(KF_p_r.processNoiseCov, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(1));
	/*KF_p_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);*/

	cv::setIdentity(KF_p_r.measurementMatrix);
	cv::setIdentity(KF_p_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_p_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_p_r.errorCovPost, cv::Scalar::all(.1)); 
}

cv::Point2f kalman_correct_point_p_r(cv::Point pt_pos_r, cv::Point pt_pos_r_old, cv::Point pt_vel_old) {
	cv::Mat prediction = KF_p_r.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_p_r(0) = pt_pos_r.x;
	measurement_p_r(1) = pt_pos_r.y;
	measurement_p_r(2) = pt_pos_r.x - pt_pos_r_old.x;
	measurement_p_r(3) = pt_pos_r.y - pt_pos_r_old.y;
	measurement_p_r(4) = measurement_p_r(2) - pt_vel_old.x;
	measurement_p_r(5) = measurement_p_r(3) - pt_vel_old.y;

	cv::Mat estimated = KF_p_r.correct(measurement_p_r);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}

cv::KalmanFilter KF_e_l (4,4,0);
cv::Mat_<float> measurement_e_l (4,1);

void init_kalman_point_e_l(cv::Point pt_pos_l) {
	KF_e_l.statePre.at<float>(0) = pt_pos_l.x;
	KF_e_l.statePre.at<float>(1) = pt_pos_l.y;
	KF_e_l.statePre.at<float>(2) = 0;
	KF_e_l.statePre.at<float>(3) = 0;

	KF_e_l.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_e_l.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);
	cv::setIdentity(KF_e_l.measurementMatrix);
	cv::setIdentity(KF_e_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_e_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_e_l.errorCovPost, cv::Scalar::all(.1)); 
}

cv::Point2f kalman_correct_point_e_l(cv::Point pt_pos_l, cv::Point pt_pos_l_old) {
	cv::Mat prediction = KF_e_l.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_e_l(0) = pt_pos_l.x;
	measurement_e_l(1) = pt_pos_l.y;
	measurement_e_l(2) = pt_pos_l.x - pt_pos_l_old.x;
	measurement_e_l(3) = pt_pos_l.y - pt_pos_l_old.y;

	cv::Mat estimated = KF_e_l.correct(measurement_e_l);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}

cv::KalmanFilter KF_e_r (4,4,0);
cv::Mat_<float> measurement_e_r (4,1);

void init_kalman_point_e_r(cv::Point pt_pos_r) {
	KF_e_r.statePre.at<float>(0) = pt_pos_r.x;
	KF_e_r.statePre.at<float>(1) = pt_pos_r.y;
	KF_e_r.statePre.at<float>(2) = 0;
	KF_e_r.statePre.at<float>(3) = 0;

	KF_e_r.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_e_r.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);
	cv::setIdentity(KF_e_r.measurementMatrix);
	cv::setIdentity(KF_e_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_e_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_e_r.errorCovPost, cv::Scalar::all(.1)); 
}

cv::Point2f kalman_correct_point_e_r(cv::Point pt_pos_r, cv::Point pt_pos_r_old) {
	cv::Mat prediction = KF_e_r.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_e_r(0) = pt_pos_r.x;
	measurement_e_r(1) = pt_pos_r.y;
	measurement_e_r(2) = pt_pos_r.x - pt_pos_r_old.x;
	measurement_e_r(3) = pt_pos_r.y - pt_pos_r_old.y;

	cv::Mat estimated = KF_e_r.correct(measurement_e_r);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}

cv::KalmanFilter KF_ce_l(6, 6, 0);
cv::Mat_<float> measurement_ce_l(6,1);

void init_kalman_ce_l(std::vector<double> vec) {

	KF_ce_l.statePre.at<float>(0) = vec[0];
	KF_ce_l.statePre.at<float>(1) = vec[1];
	KF_ce_l.statePre.at<float>(2) = vec[2];
	KF_ce_l.statePre.at<float>(3) = 0;
	KF_ce_l.statePre.at<float>(4) = 0;
	KF_ce_l.statePre.at<float>(5) = 0;


	KF_ce_l.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ce_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ce_l.measurementMatrix);
	cv::setIdentity(KF_ce_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ce_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ce_l.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_ce_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ce_l.predict();
	measurement_ce_l(0) = vec[0];
	measurement_ce_l(1) = vec[1];
	measurement_ce_l(2) = vec[2];
	measurement_ce_l(3) = vec[0] - old[0];
	measurement_ce_l(4) = vec[1] - old[1];
	measurement_ce_l(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ce_l.correct(measurement_ce_l);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

cv::KalmanFilter KF_ce_r(6, 6, 0);
cv::Mat_<float> measurement_ce_r(6,1);

void init_kalman_ce_r(std::vector<double> vec) {

	KF_ce_r.statePre.at<float>(0) = vec[0];
	KF_ce_r.statePre.at<float>(1) = vec[1];
	KF_ce_r.statePre.at<float>(2) = vec[2];
	KF_ce_r.statePre.at<float>(3) = 0;
	KF_ce_r.statePre.at<float>(4) = 0;
	KF_ce_r.statePre.at<float>(5) = 0;


	KF_ce_r.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ce_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ce_r.measurementMatrix);
	cv::setIdentity(KF_ce_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ce_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ce_r.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_ce_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ce_r.predict();
	measurement_ce_r(0) = vec[0];
	measurement_ce_r(1) = vec[1];
	measurement_ce_r(2) = vec[2];
	measurement_ce_r(3) = vec[0] - old[0];
	measurement_ce_r(4) = vec[1] - old[1];
	measurement_ce_r(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ce_r.correct(measurement_ce_r);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

cv::KalmanFilter KF_ep_l(6, 6, 0);
cv::Mat_<float> measurement_ep_l(6,1);

void init_kalman_ep_l(std::vector<double> vec) {

	KF_ep_l.statePre.at<float>(0) = vec[0];
	KF_ep_l.statePre.at<float>(1) = vec[1];
	KF_ep_l.statePre.at<float>(2) = vec[2];
	KF_ep_l.statePre.at<float>(3) = 0;
	KF_ep_l.statePre.at<float>(4) = 0;
	KF_ep_l.statePre.at<float>(5) = 0;


	KF_ep_l.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ep_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ep_l.measurementMatrix);
	cv::setIdentity(KF_ep_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ep_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ep_l.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_ep_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ep_l.predict();
	measurement_ep_l(0) = vec[0];
	measurement_ep_l(1) = vec[1];
	measurement_ep_l(2) = vec[2];
	measurement_ep_l(3) = vec[0] - old[0];
	measurement_ep_l(4) = vec[1] - old[1];
	measurement_ep_l(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ep_l.correct(measurement_ep_l);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

cv::KalmanFilter KF_ep_r(6, 6, 0);
cv::Mat_<float> measurement_ep_r(6,1);

void init_kalman_ep_r(std::vector<double> vec) {

	KF_ep_r.statePre.at<float>(0) = vec[0];
	KF_ep_r.statePre.at<float>(1) = vec[1];
	KF_ep_r.statePre.at<float>(2) = vec[2];
	KF_ep_r.statePre.at<float>(3) = 0;
	KF_ep_r.statePre.at<float>(4) = 0;
	KF_ep_r.statePre.at<float>(5) = 0;


	KF_ep_r.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ep_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ep_r.measurementMatrix);
	cv::setIdentity(KF_ep_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ep_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ep_r.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_ep_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ep_r.predict();
	measurement_ep_r(0) = vec[0];
	measurement_ep_r(1) = vec[1];
	measurement_ep_r(2) = vec[2];
	measurement_ep_r(3) = vec[0] - old[0];
	measurement_ep_r(4) = vec[1] - old[1];
	measurement_ep_r(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ep_r.correct(measurement_ep_r);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

cv::KalmanFilter KF_cp_l(6, 6, 0);
cv::Mat_<float> measurement_cp_l(6,1);

void init_kalman_cp_l(std::vector<double> vec) {

	KF_cp_l.statePre.at<float>(0) = vec[0];
	KF_cp_l.statePre.at<float>(1) = vec[1];
	KF_cp_l.statePre.at<float>(2) = vec[2];
	KF_cp_l.statePre.at<float>(3) = 0;
	KF_cp_l.statePre.at<float>(4) = 0;
	KF_cp_l.statePre.at<float>(5) = 0;


	KF_cp_l.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_cp_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_cp_l.measurementMatrix);
	cv::setIdentity(KF_cp_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_cp_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_cp_l.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_cp_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_cp_l.predict();
	measurement_cp_l(0) = vec[0];
	measurement_cp_l(1) = vec[1];
	measurement_cp_l(2) = vec[2];
	measurement_cp_l(3) = vec[0] - old[0];
	measurement_cp_l(4) = vec[1] - old[1];
	measurement_cp_l(5) = vec[2] - old[2];

	cv::Mat estimated = KF_cp_l.correct(measurement_cp_l);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

cv::KalmanFilter KF_cp_r(6, 6, 0);
cv::Mat_<float> measurement_cp_r(6,1);

void init_kalman_cp_r(std::vector<double> vec) {

	KF_cp_r.statePre.at<float>(0) = vec[0];
	KF_cp_r.statePre.at<float>(1) = vec[1];
	KF_cp_r.statePre.at<float>(2) = vec[2];
	KF_cp_r.statePre.at<float>(3) = 0;
	KF_cp_r.statePre.at<float>(4) = 0;
	KF_cp_r.statePre.at<float>(5) = 0;


	KF_cp_r.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_cp_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_cp_r.measurementMatrix);
	cv::setIdentity(KF_cp_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_cp_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_cp_r.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_cp_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_cp_r.predict();
	measurement_cp_r(0) = vec[0];
	measurement_cp_r(1) = vec[1];
	measurement_cp_r(2) = vec[2];
	measurement_cp_r(3) = vec[0] - old[0];
	measurement_cp_r(4) = vec[1] - old[1];
	measurement_cp_r(5) = vec[2] - old[2];

	cv::Mat estimated = KF_cp_r.correct(measurement_cp_r);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}
