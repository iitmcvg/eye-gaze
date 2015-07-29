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

#include "faceDetection.h"
#include "pupilDetection.h"
#include "kalmanFilters.h"
#include "util.h"

#include "pupilCdf.h"
#include "kmeansUtils.h"
#include "gazeComputation.h"

using namespace dlib;
using namespace std;

void preprocessROI(cv::Mat& roi_eye) {
    GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
    equalizeHist( roi_eye, roi_eye );
}

int main(int argc, char** argv) {
    try	{

        cv::VideoCapture cap(0);
        image_window win;

        FaceFeatures *face_features = new FaceFeatures();
        FaceData *face_data = new FaceData();
        FacePose *face_pose = new FacePose();

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;

        std::vector<double> vec_ce_pos_l(3), vec_ce_vel_l(3), vec_ce_pos_l_old(3), vec_ce_vel_l_old(3), vec_ce_kalman_l(3);
        std::vector<double> vec_ep_pos_l(3), vec_ep_vel_l(3), vec_ep_pos_l_old(3), vec_ep_vel_l_old(3), vec_ep_kalman_l(3);
        std::vector<double> vec_cp_pos_l(3), vec_cp_vel_l(3), vec_cp_pos_l_old(3), vec_cp_vel_l_old(3), vec_cp_kalman_l(3);

        std::vector<double> vec_ce_pos_r(3), vec_ce_vel_r(3), vec_ce_pos_r_old(3), vec_ce_vel_r_old(3), vec_ce_kalman_r(3);
        std::vector<double> vec_ep_pos_r(3), vec_ep_vel_r(3), vec_ep_pos_r_old(3), vec_ep_vel_r_old(3), vec_ep_kalman_r(3);
        std::vector<double> vec_cp_pos_r(3), vec_cp_vel_r(3), vec_cp_pos_r_old(3), vec_cp_vel_r_old(3), vec_cp_kalman_r(3);

        std::vector<double> center_eye_proj(3);
        std::vector<double> vec_cp_kalman_avg(3);

        std::vector<std::vector<double> > vec_kmeans_centers_l;
        std::vector<float> vec_kmeans_data_l;

        double Cf_left, Cf_right, mag_nor = 12.0, alpha = 30.0;

        //TODO : Initialize all vectors to [0, 0, 0];
        vec_ce_pos_l[0] = 0;vec_ce_pos_l[1] = 0;vec_ce_pos_l[2] = 0;
        vec_ce_pos_l_old[0] = 0;vec_ce_pos_l_old[1] = 0;vec_ce_pos_l_old[2] = 0;

        vec_ce_pos_r[0] = 0;vec_ce_pos_r[1] = 0;vec_ce_pos_r[2] = 0;
        vec_ce_pos_r_old[0] = 0;vec_ce_pos_r_old[1] = 0;vec_ce_pos_r_old[2] = 0;


        vec_ep_pos_l[0] = 0;vec_ep_pos_l[1] = 0;vec_ep_pos_l[2] = 0;
        vec_ep_pos_l_old[0] = 0;vec_ep_pos_l_old[1] = 0;vec_ep_pos_l_old[2] = 0;

        vec_ep_pos_r[0] = 0;vec_ep_pos_r[1] = 0;vec_ep_pos_r[2] = 0;
        vec_ep_pos_r_old[0] = 0;vec_ep_pos_r_old[1] = 0;vec_ep_pos_r_old[2] = 0;


        vec_cp_pos_l[0] = 0;vec_cp_pos_l[1] = 0;vec_cp_pos_l[2] = 0;
        vec_cp_pos_l_old[0] = 0;vec_cp_pos_l_old[1] = 0;vec_cp_pos_l_old[2] = 0;

        vec_cp_pos_r[0] = 0;vec_cp_pos_r[1] = 0;vec_cp_pos_r[2] = 0;
        vec_cp_pos_r_old[0] = 0;vec_cp_pos_r_old[1] = 0;vec_cp_pos_r_old[2] = 0;


        cv::Point pt_p_pos_l(0,0), pt_p_vel_l(0,0), pt_p_pos_l_old(0,0), pt_p_kalman_l(0,0), pt_p_vel_l_old(0,0);
        cv::Point pt_e_pos_l(0,0), pt_e_vel_l(0,0), pt_e_pos_l_old(0,0), pt_e_kalman_l(0,0);

        cv::Point pt_p_pos_r(0,0), pt_p_vel_r(0,0), pt_p_pos_r_old(0,0), pt_p_kalman_r(0,0), pt_p_vel_r_old(0,0);
        cv::Point pt_e_pos_r(0,0), pt_e_vel_r(0,0), pt_e_pos_r_old(0,0), pt_e_kalman_r(0,0);

        cv::Rect rect1, rect2;

        cv::Mat frame, frame_clr, temp, temp2, roi1, roi1_temp, roi2, roi1_clr, roi2_clr ,roi1_clr_temp;
        int k_pt_e_l = 0, k_pt_p_l = 0, k_vec_ce_l = 0, k_vec_cp_l = 0, k_vec_ep_l = 0;
        int k_pt_e_r = 0, k_pt_p_r = 0, k_vec_ce_r = 0, k_vec_cp_r = 0, k_vec_ep_r = 0;

        while(!win.is_closed()) {
            cap >> frame_clr;
            cv::flip(frame_clr, frame_clr, 1);
            cv::cvtColor(frame_clr, frame, CV_BGR2GRAY);

            cv_image<unsigned char> cimg_gray(frame);
            cv_image<bgr_pixel> cimg_clr(frame_clr);

            std::vector<rectangle> faces = detector(cimg_gray);

            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg_gray, faces[i]));

            if(shapes.size() == 0) {
                std::cout<<"zero faces"<<std::endl;
                k_pt_p_l=0;
                k_pt_e_l=0;
                k_vec_ce_l=0;
                k_vec_ep_l=0;
                k_pt_p_r=0;
                k_pt_e_r=0;
                k_vec_ce_r=0;
                k_vec_ep_r=0;
            }
            else{
                //TODO : Initialize the variables used in the Kalman filter
                pt_p_pos_l_old = pt_p_pos_l;
                pt_p_vel_l_old = pt_p_vel_l;
                pt_e_pos_l_old = pt_e_pos_l;

                pt_p_pos_r_old = pt_p_pos_r;
                pt_p_vel_r_old = pt_p_vel_r;
                pt_e_pos_r_old = pt_e_pos_r;


                vec_ce_pos_l_old = vec_ce_pos_l;
                vec_ep_pos_l_old = vec_ep_pos_l;
                vec_cp_pos_l_old = vec_cp_pos_l;

                vec_ce_pos_r_old = vec_ce_pos_r;
                vec_ep_pos_r_old = vec_ep_pos_r;
                vec_cp_pos_r_old = vec_cp_pos_r;

                dlib::full_object_detection shape = shapes[0];

                face_features->assign(cv::Point(0,0),
                        get_mid_point(cv::Point(shape.part(42).x(), shape.part(42).y()),
                            cv::Point(shape.part(45).x(), shape.part(45).y())),
                        get_mid_point(cv::Point(shape.part(36).x(), shape.part(36).y()),
                            cv::Point(shape.part(39).x(), shape.part(39).y())),
                        cv::Point(shape.part(30).x(), shape.part(30).y()), 
                        get_mid_point(cv::Point(shape.part(48).x(), shape.part(48).y()),
                            cv::Point(shape.part(54).x(), shape.part(54).y())));

                face_data->assign(face_features);

                face_pose->assign(face_features, face_data);

                //Cf_left = get_conversion_factor(shape, face_pose, alpha, 0);
                //Cf_right = get_conversion_factor(shape, face_pose, alpha, 1);

                std::vector<cv::Point> vec_pts_left_eye(0), vec_pts_right_eye(0);

                for(int j=42;j<=47;j++) 
                    vec_pts_left_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));

                for(int j=36;j<=41;j++) 
                    vec_pts_right_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));

                rect1 = cv::boundingRect(vec_pts_left_eye);
                rect2 = cv::boundingRect(vec_pts_right_eye);

                blow_up_rect(rect1, 2.0);
                blow_up_rect(rect2, 2.0);

                roi1 = frame(rect1);
                roi2 = frame(rect2);
                roi1.copyTo(roi1_temp);

                preprocessROI(roi1);
                preprocessROI(roi2);

                pt_p_pos_l = get_pupil_coordinates(roi1, rect1);

                double mag_CP = 12.0, mag_LR = 30.0, mag_CR = 12.0, mag_CM = 12.0, theta = 60*3.14/180.0;

                compute_eye_gaze (face_pose, shape, pt_p_pos_l, mag_CP, mag_LR, mag_CR, mag_CM, theta, 1, vec_cp_pos_l);
                draw_eye_gaze(pt_p_pos_l, vec_cp_pos_l, rect1, frame_clr);

/*              roi1_clr = frame_clr(rect1);
                roi2_clr = frame_clr(rect2);

                roi1_clr.copyTo(roi1_clr_temp);

                kmeans_array_generate(roi1_clr, vec_kmeans_data_l, 0);
                cout<<vec_kmeans_data_l.size()<<std::endl;

                std::vector<int> vec_kmeans_labels_l(vec_kmeans_data_l.size());
                cv::kmeans(cv::Mat(vec_kmeans_data_l).reshape(1, vec_kmeans_data_l.size()), 3, vec_kmeans_labels_l, cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
                        3, cv::KMEANS_PP_CENTERS);//, vec_kmeans_centers_l);

                //TODO : Compute current values and correct values using Kalman filter
                pt_e_pos_l = get_mid_point(cv::Point(shape.part(42).x(), shape.part(42).y()),cv::Point(shape.part(45).x(), shape.part(45).y()));
                pt_e_pos_r = get_mid_point(cv::Point(shape.part(36).x(), shape.part(36).y()),cv::Point(shape.part(39).x(), shape.part(39).y()));


                cv::circle(frame_clr, pt_e_pos_l, 1, cv::Scalar(255,0,0), 1, 4, 0);
                cv::circle(frame_clr, pt_e_pos_r, 1, cv::Scalar(255,0,0), 1, 4, 0);

                pt_e_pos_l.x -= rect1.x;
                pt_e_pos_l.y -= rect1.y;
                pt_e_vel_l.x = pt_e_pos_l.x - pt_e_pos_l_old.x;
                pt_e_vel_l.y = pt_e_pos_l.y - pt_e_pos_l_old.y;

                pt_e_pos_r.x -= rect2.x;
                pt_e_pos_r.y -= rect2.y;
                pt_e_vel_r.x = pt_e_pos_r.x - pt_e_pos_r_old.x;
                pt_e_vel_r.y = pt_e_pos_r.y - pt_e_pos_r_old.y;

                if(k_pt_e_l == 0) {
                    pt_e_pos_l_old.x = 0;
                    pt_e_pos_l_old.y = 0;
                    init_kalman_point_e_l(pt_e_pos_l);
                    ++k_pt_e_l;
                }

                if(k_pt_e_r == 0) {
                    pt_e_pos_r_old.x = 0;
                    pt_e_pos_r_old.y = 0;
                    init_kalman_point_e_r(pt_e_pos_r);
                    ++k_pt_e_r;
                }

                pt_e_kalman_l = kalman_correct_point_e_l(pt_e_pos_l, pt_e_pos_l_old);
                pt_e_kalman_r = kalman_correct_point_e_r(pt_e_pos_r, pt_e_pos_r_old);

                std::cout<<"Point E - l "<<pt_e_kalman_l.x<<" "<<pt_e_kalman_l.y<<endl;
                std::cout<<"Point E - l "<<pt_e_kalman_r.x<<" "<<pt_e_kalman_r.y<<endl;

                pt_p_pos_l = get_pupil_coordinates(roi1, rect1);
                pt_p_vel_l.x = pt_p_pos_l.x - pt_p_pos_l_old.x;
                pt_p_vel_l.y = pt_p_pos_l.y - pt_p_pos_l_old.y;

                pt_p_pos_r = get_pupil_coordinates(roi2, rect2);
                pt_p_vel_r.x = pt_p_pos_r.x - pt_p_pos_r_old.x;
                pt_p_vel_r.y = pt_p_pos_r.y - pt_p_pos_r_old.y;

                if(k_pt_p_l == 0) {
                    pt_p_pos_l_old.x = 0;
                    pt_p_pos_l_old.y = 0;
                    init_kalman_point_p_l(pt_p_pos_l);
                    ++k_pt_p_l;
                }

                if(k_pt_p_r == 0) {
                    pt_p_pos_r_old.x = 0;
                    pt_p_pos_r_old.y = 0;
                    init_kalman_point_p_r(pt_p_pos_r);
                    ++k_pt_p_r;
                }

                pt_p_kalman_l = kalman_correct_point_p_l(pt_p_pos_l, pt_p_pos_l_old, pt_p_vel_l);
                pt_p_kalman_r = kalman_correct_point_p_r(pt_p_pos_r, pt_p_pos_r_old, pt_p_vel_r);

                if(!is_point_in_mat(pt_p_kalman_l, roi1)) {
                    k_pt_p_l=0;
                    k_pt_e_l=0;
                    k_vec_ce_l=0;
                    k_vec_ep_l=0;
                }

                if(!is_point_in_mat(pt_p_kalman_r, roi1)) {
                    k_pt_p_r=0;
                    k_pt_e_r=0;
                    k_vec_ce_r=0;
                    k_vec_ep_r=0;
                }

                std::cout<<"Point P "<<pt_p_kalman_l.x<<" "<<pt_p_kalman_l.y<<endl;
                std::cout<<"Point P "<<pt_p_kalman_r.x<<" "<<pt_p_kalman_r.y<<endl;

                vec_ce_pos_l[0] = face_pose->normal[0];
                vec_ce_pos_l[1] = face_pose->normal[1];
                vec_ce_pos_l[2] = face_pose->normal[2];

                vec_ce_pos_r[0] = face_pose->normal[0];
                vec_ce_pos_r[1] = face_pose->normal[1];
                vec_ce_pos_r[2] = face_pose->normal[2];

                vec_ce_vel_l[0] = vec_ce_pos_l[0] - vec_ce_pos_l_old[0];
                vec_ce_vel_l[1] = vec_ce_pos_l[1] - vec_ce_pos_l_old[1];
                vec_ce_vel_l[2] = vec_ce_pos_l[2] - vec_ce_pos_l_old[2];

                vec_ce_vel_r[0] = vec_ce_pos_r[0] - vec_ce_pos_r_old[0];
                vec_ce_vel_r[1] = vec_ce_pos_r[1] - vec_ce_pos_r_old[1];
                vec_ce_vel_r[2] = vec_ce_pos_r[2] - vec_ce_pos_r_old[2];

                if(k_vec_ce_l == 0) {
                    vec_ce_pos_l_old[0] = 0;vec_ce_pos_l_old[1] = 0;vec_ce_pos_l_old[2] = 0;
                    init_kalman_ce_l(vec_ce_pos_l);
                    ++k_vec_ce_l;
                }

                if(k_vec_ce_r == 0) {
                    vec_ce_pos_r_old[0] = 0;vec_ce_pos_r_old[1] = 0;vec_ce_pos_r_old[2] = 0;
                    init_kalman_ce_r(vec_ce_pos_r);
                    ++k_vec_ce_r;
                }

                kalman_predict_correct_ce_l(vec_ce_pos_l, vec_ce_pos_l_old, vec_ce_kalman_l);
                kalman_predict_correct_ce_r(vec_ce_pos_r, vec_ce_pos_r_old, vec_ce_kalman_r);

                make_unit_vector(vec_ce_pos_l, vec_ce_pos_l);
                make_unit_vector(vec_ce_kalman_l, vec_ce_kalman_l);
                std::cout<<"Vector CE "<<vec_ce_kalman_l[0]<<" "<<vec_ce_kalman_l[1]<<" "<<vec_ce_kalman_l[2]<<endl;

                make_unit_vector(vec_ce_pos_r, vec_ce_pos_r);
                make_unit_vector(vec_ce_kalman_r, vec_ce_kalman_r);
                std::cout<<"Vector CE "<<vec_ce_kalman_r[0]<<" "<<vec_ce_kalman_r[1]<<" "<<vec_ce_kalman_r[2]<<endl;

                vec_ep_pos_l[0] = pt_p_kalman_l.x - pt_e_kalman_l.x;
                vec_ep_pos_l[1] = pt_p_kalman_l.y - pt_e_kalman_l.y;
                vec_ep_pos_l[2] = 0.0;

                vec_ep_pos_r[0] = pt_p_kalman_r.x - pt_e_kalman_r.x;
                vec_ep_pos_r[1] = pt_p_kalman_r.y - pt_e_kalman_r.y;
                vec_ep_pos_r[2] = 0.0;

                vec_ep_pos_l[0] = pt_p_pos_l.x - pt_e_pos_l.x;
                vec_ep_pos_l[1] = pt_p_pos_l.y - pt_e_pos_l.y;
                vec_ep_pos_l[2] = 0.0;

                vec_ep_pos_r[0] = pt_p_pos_r.x - pt_e_pos_r.x;
                vec_ep_pos_r[1] = pt_p_pos_r.y - pt_e_pos_r.y;
                vec_ep_pos_r[2] = 0.0;

                if(k_vec_ep_l == 0) {
                    vec_ep_pos_l_old[0] = 0;
                    vec_ep_pos_l_old[1] = 0;
                    vec_ep_pos_l_old[2] = 0;
                    init_kalman_ep_l(vec_ep_pos_l);
                    ++k_vec_ep_l;
                }

                if(k_vec_ep_r == 0) {
                    vec_ep_pos_r_old[0] = 0;
                    vec_ep_pos_r_old[1] = 0;
                    vec_ep_pos_r_old[2] = 0;
                    init_kalman_ep_r(vec_ep_pos_r);
                    ++k_vec_ep_r;
                }

                kalman_predict_correct_ep_l(vec_ep_pos_l, vec_ep_pos_l_old, vec_ep_kalman_l);
                kalman_predict_correct_ep_r(vec_ep_pos_r, vec_ep_pos_r_old, vec_ep_kalman_r);

                vec_cp_pos_l[0] = (mag_nor*Cf_left*vec_ce_kalman_l[0]) + vec_ep_pos_l[0];
                vec_cp_pos_l[1] = (mag_nor*Cf_left*vec_ce_kalman_l[1]) + vec_ep_pos_l[1];
                vec_cp_pos_l[2] = (mag_nor*Cf_left*vec_ce_kalman_l[2]) + vec_ep_pos_l[2];

                vec_cp_pos_r[0] = (mag_nor*Cf_right*vec_ce_kalman_r[0]) + vec_ep_pos_r[0];
                vec_cp_pos_r[1] = (mag_nor*Cf_right*vec_ce_kalman_r[1]) + vec_ep_pos_r[1];
                vec_cp_pos_r[2] = (mag_nor*Cf_right*vec_ce_kalman_r[2]) + vec_ep_pos_r[2];

                vec_cp_vel_l[0] = vec_cp_pos_l[0] - vec_cp_pos_l_old[0];
                vec_cp_vel_l[1] = vec_cp_pos_l[1] - vec_cp_pos_l_old[1];
                vec_cp_vel_l[2] = vec_cp_pos_l[2] - vec_cp_pos_l_old[2];

                vec_cp_vel_r[0] = vec_cp_pos_r[0] - vec_cp_pos_r_old[0];
                vec_cp_vel_r[1] = vec_cp_pos_r[1] - vec_cp_pos_r_old[1];
                vec_cp_vel_r[2] = vec_cp_pos_r[2] - vec_cp_pos_r_old[2];

                if(k_vec_cp_l == 0) {
                    vec_cp_pos_l_old[0] = 0;
                    vec_cp_pos_l_old[1] = 0;
                    vec_cp_pos_l_old[2] = 0;
                    init_kalman_cp_l(vec_cp_pos_l);
                    ++k_vec_cp_l;
                }

                if(k_vec_cp_r == 0) {
                    vec_cp_pos_r_old[0] = 0;
                    vec_cp_pos_r_old[1] = 0;
                    vec_cp_pos_r_old[2] = 0;
                    init_kalman_cp_r(vec_cp_pos_r);
                    ++k_vec_cp_r;
                }

                kalman_predict_correct_cp_l(vec_cp_pos_l, vec_cp_pos_l_old, vec_cp_kalman_l);
                kalman_predict_correct_cp_r(vec_cp_pos_r, vec_cp_pos_r_old, vec_cp_kalman_r);

                //make_unit_vector(vec_cp_pos_l, vec_cp_pos_l);
                //make_unit_vector(vec_cp_pos_r, vec_cp_pos_r);

                std::cout<<"Vector CP "<<vec_cp_kalman_l[0]<<" "<<vec_cp_kalman_l[1]<<" "<<vec_cp_kalman_l[2]<<endl;
                std::cout<<"Vector CP "<<vec_cp_kalman_r[0]<<" "<<vec_cp_kalman_r[1]<<" "<<vec_cp_kalman_r[2]<<endl;
                
                make_unit_vector(vec_cp_kalman_l, vec_cp_kalman_l);
                make_unit_vector(vec_cp_kalman_r, vec_cp_kalman_r);

                vec_cp_kalman_avg[0] = (vec_cp_kalman_l[0] + vec_cp_kalman_r[0])/2.0;
                vec_cp_kalman_avg[1] = (vec_cp_kalman_l[1] + vec_cp_kalman_r[1])/2.0;
                vec_cp_kalman_avg[2] = (vec_cp_kalman_l[2] + vec_cp_kalman_r[2])/2.0;		

                draw_eye_gaze(pt_p_kalman_l, vec_cp_kalman_avg, rect1, frame_clr);				
                //draw_eye_gaze(pt_p_kalman_r, vec_cp_kalman_avg, rect2, frame_clr);
                cv::line(frame_clr, cv::Point(pt_p_kalman_r.x + rect2.x, pt_p_kalman_r.y + rect2.y), cv::Point(pt_p_kalman_r.x + rect2.x + vec_ep_pos_r[0], pt_p_kalman_r.y + rect2.y + vec_ep_pos_r[1]), cv::Scalar(255, 255, 255), 1);

                draw_facial_normal(frame_clr, shape, vec_ce_kalman_l, 5*mag_nor);*/
            }
            win.clear_overlay();
            win.set_image(cimg_clr);
            //win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(serialization_error& e) {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e) {
        cout << e.what() << endl;
    }
    return 0;
}