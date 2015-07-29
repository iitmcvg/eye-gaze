#ifndef GAZE_COMPUTATION_H
#define GAZE_COMPUTATION_H

double get_conversion_factor (dlib::full_object_detection shape, FacePose* face_pose, double magnitude_normal, int mode);
void compute_vec_LR (cv::Point p1, cv::Point p2, FacePose* face_pose, std::vector<double>& LR);
void get_quadratic_solution (std::vector<double> coeff, double& solution, int mode);
void get_quadratic_equation (std::vector<double> coeff, std::vector<double>& quad_eqn);
void solve(std::vector<double> coeff_1, double const_1, std::vector<double> coeff_2, double const_2, double mag, std::vector<double>& vec, int mode);
void get_section(cv::Point p1, cv::Point p2, cv::Point pupil, double& Y1, double& Y2, double& h);
void compute_vec_CP(cv::Point p1, cv::Point p2, cv::Point pupil, FacePose* face_pose, std::vector<double> vec_CR_u, double mag_CR, std::vector<double> vec_LR_u, double mag_LR, std::vector<double> vec_UD_u, double mag_CP, std::vector<double>& vec_CP, double S2R);
void compute_eye_gaze (FacePose* face_pose, dlib::full_object_detection shape, cv::Point pupil, double mag_CP, double mag_LR, double mag_CR, double mag_CM, double theta, int mode, std::vector<double>& vec_CP);

#endif