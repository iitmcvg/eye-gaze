#ifndef KALMAN_FILTERS_H
#define KALMAN_FILTERS_H

void init_kalman_point_p_l(cv::Point pt_pos_l);
cv::Point2f kalman_correct_point_p_l(cv::Point pt_pos_l, cv::Point pt_pos_l_old, cv::Point pt_vel_old);

void init_kalman_point_p_r(cv::Point pt_pos_r);
cv::Point2f kalman_correct_point_p_r(cv::Point pt_pos_r, cv::Point pt_pos_r_old, cv::Point pt_vel_old);

void init_kalman_point_e_l(cv::Point pt_pos_l);
cv::Point2f kalman_correct_point_e_l(cv::Point pt_pos_l, cv::Point pt_pos_l_old);

void init_kalman_point_e_r(cv::Point pt_pos_r);
cv::Point2f kalman_correct_point_e_r(cv::Point pt_pos_r, cv::Point pt_pos_r_old);

void init_kalman_ce_l(std::vector<double> vec);
void kalman_predict_correct_ce_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred);

void init_kalman_ce_r(std::vector<double> vec);
void kalman_predict_correct_ce_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred);

void init_kalman_ep_l(std::vector<double> vec);
void kalman_predict_correct_ep_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred);

void init_kalman_ep_r(std::vector<double> vec);
void kalman_predict_correct_ep_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred);

void init_kalman_cp_l(std::vector<double> vec);
void kalman_predict_correct_cp_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred);

void init_kalman_cp_r(std::vector<double> vec);
void kalman_predict_correct_cp_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred);

#endif
