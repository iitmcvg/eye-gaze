#ifndef PUPIL_DETECTION_H
#define PUPIL_DETECTION_H

cv::Point unscale_point(cv::Point p, cv::Rect origSize);
void scale_standard_size(const cv::Mat &src,cv::Mat &dst);
cv::Mat compute_x_gradient(const cv::Mat &mat);
double compute_threshold(const cv::Mat &mat, double stdDevFactor);
cv::Mat get_matrix_magnitude(const cv::Mat &matX, const cv::Mat &matY);
bool check_point_in_mat(cv::Point p,int rows,int cols);
bool is_point_in_mat(const cv::Point &np, const cv::Mat &mat);
cv::Mat remove_edges(cv::Mat &mat);
void check_pupil(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out);
cv::Point get_pupil_coordinates(cv::Mat eye_mat,cv::Rect eye);
void draw_eye_gaze(cv::Point pt, std::vector<double> vec_gaze, cv::Rect roi_eye, cv::Mat& img, int scale);
double get_z_component(double Cf_left, cv::Point pt_p_kalman, cv::Point pt_e_kalman, std::vector<double> vec_ce_kalman);
void retrace_eye_center(cv::Point& pt_e_pos, double normal[3], double mag);

#endif
