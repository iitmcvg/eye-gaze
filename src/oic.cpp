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

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>


#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

using namespace dlib;
using namespace std;

///////////////////////Defining keypress'
#define DOWN_KEY XK_Down
#define UP_KEY XK_Up
#define RIGHT_KEY XK_Right
#define LEFT_KEY XK_Left

XKeyEvent createKeyEvent(Display *display, Window &win, Window &winRoot, bool press, int keycode, int modifiers)
{
    XKeyEvent event;

    event.display     = display;
    event.window      = win;
    event.root        = winRoot;
    event.subwindow   = None;
    event.time        = CurrentTime;
    event.x           = 1;
    event.y           = 1;
    event.x_root      = 1;
    event.y_root      = 1;
    event.same_screen = True;
    event.keycode     = XKeysymToKeycode(display, keycode);
    event.state       = modifiers;

    if(press)
        event.type = KeyPress;
    else
        event.type = KeyRelease;

    return event;
}

void simulate_key_press(int key_code) {
// Obtain the X11 display.
    Display *display = XOpenDisplay(0);    

// Get the root window for the current display.
    Window winRoot = XDefaultRootWindow(display);

// Find the window which has the current keyboard focus.
    Window winFocus;
    int    revert;
    XGetInputFocus(display, &winFocus, &revert);

// Send a fake key press event to the window.
    XKeyEvent event = createKeyEvent(display, winFocus, winRoot, true, key_code, 0);
    XSendEvent(event.display, event.window, True, KeyPressMask, (XEvent *)&event);

// Send a fake key release event to the window.
    event = createKeyEvent(display, winFocus, winRoot, false, key_code, 0);
    XSendEvent(event.display, event.window, True, KeyPressMask, (XEvent *)&event);

// Done.
    XCloseDisplay(display);
}


float R[16];

void matMul(float Rx[3][3], float Ry[3][3], float Rz[3][3], float Rt[3][3]) {
    float temp[3][3];
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            temp[i][j] = 0.0;
            for(int k=0; k<3; ++k) {
                temp[i][j] += Rx[i][k]*Ry[k][j];
            }
        }
    }
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            Rt[i][j] = 0.0;
            for(int k=0; k<3; ++k) {
                Rt[i][j] += temp[i][k]*Rz[k][j];
            }
        }
    }
}

void constructRot(double nx, double ny, double nz) {
    double alpha = acos(nx);
    double beta = acos(ny);
    double gamma = acos(nz);

    float Rx[3][3] = {
        { 
            1, 0, 0 
        },
        {
            0, cos(alpha), sin(alpha)
        },
        {
            0, -sin(alpha), cos(alpha)
        }
    };

    float Ry[3][3] = {
        {
            cos(beta), 0, -sin(beta)
        },
        {
            0, 1, 0
        },
        {
            sin(beta), 0, cos(beta)
        }
    };

    float Rz[3][3] = {
        {
            cos(gamma), sin(gamma), 0
        },
        {
            -sin(gamma), cos(gamma), 0
        },
        {
            0, 0, 1
        }
    };

    float Rt[3][3];

    matMul(Rx, Ry, Rz, Rt);
    int i=0;
    for(int j=0; j<3; ++j) {
        for(int k=0; k<3; ++k) {
            R[i++] = Rt[k][j];
        }
    }
}

void preprocessROI(cv::Mat& roi_eye) {
    GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
    equalizeHist( roi_eye, roi_eye );
}

std::vector<double> g_normal(3, 0);

void* startOCV(void* argv) {
    try	{
/*
        init ();
        glutDisplayFunc(display);*/

        cv::VideoCapture cap(0);
        image_window win, win1;

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

                Cf_left = get_conversion_factor(shape, face_pose, alpha, 1);
                Cf_right = get_conversion_factor(shape, face_pose, alpha, 2);

                std::vector<cv::Point> vec_pts_left_eye(0), vec_pts_right_eye(0);

                for(int j=42;j<=47;j++) 
                    vec_pts_left_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));

                for(int j=36;j<=41;j++) 
                    vec_pts_right_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));

                rect1 = cv::boundingRect(vec_pts_left_eye);
                rect2 = cv::boundingRect(vec_pts_right_eye);

                blow_up_rect(rect1, 1.5);
                blow_up_rect(rect2, 1.5);

                roi1 = frame(rect1);
                roi2 = frame(rect2);
                roi1.copyTo(roi1_temp);

                preprocessROI(roi1);
                preprocessROI(roi2);

                pt_p_pos_l = get_pupil_coordinates(roi1, rect1);

                double mag_LR = 30.0, mag_CR = 12.0, theta = 113.4*3.14/180.0;
                double dist_AB = 5.101, mag_CM = dist_AB + 8.0, mag_CP = 13.101;

                if (0/*atoi(argv[1]) == 1*/) {
                compute_eye_gaze (face_pose, shape, rect1, pt_p_pos_l, mag_CP, mag_LR, mag_CR, mag_CM, theta, 1, vec_cp_pos_l);
                std::cout<<"Yaw : "<<face_pose->yaw*180.0/3.14<<" Pitch : "<<face_pose->pitch*180.0/3.14<<" Roll : "<<face_pose->symm_x*180.0/3.14<<std::endl;
                log_vec("CP_l", vec_cp_pos_l);  
                std::cout<<"normal[0] : "<< face_pose->normal[0]<<" normal[1] : "<<face_pose->normal[1]<<" normal[2] : "<<face_pose->normal[2]<<std::endl;
                draw_eye_gaze(pt_p_pos_l, vec_cp_pos_l, rect1, frame_clr, 2);
            }
                else if (0/*atoi(argv[1]) == 2*/) {
            roi1_clr = frame_clr(rect1);
            filter_image(roi1_clr, pt_p_pos_l);
                    //cv::circle(roi1_clr, pt_e_pos_l, 1, cv::Scalar(255,0,0), 1, 4, 0);
            cv_image<bgr_pixel> roi1_img(roi1_clr);
            win1.set_image(roi1_img);
        }
                else if (1/*atoi(argv[1]) == 0*/) {
        roi1_clr = frame_clr(rect1);
        roi2_clr = frame_clr(rect2);

        roi1_clr.copyTo(roi1_clr_temp);

        pt_e_pos_l = get_mid_point(cv::Point(shape.part(42).x(), shape.part(42).y()),
            cv::Point(shape.part(45).x(), shape.part(45).y()));
        pt_e_pos_r = get_mid_point(cv::Point(shape.part(36).x(), shape.part(36).y()),
            cv::Point(shape.part(39).x(), shape.part(39).y()));

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

///////////////////////////////////////////////////////////////////////////////////Checking for direction of vector
//If it doesnt work well just check with face_normal vector
//Also change the threshold value used to check vector direction        

        if(vec_cp_kalman_avg[0]>0) {
                        std::cout<<endl<<"Right ";
                        simulate_key_press(RIGHT_KEY);
                    }
        else if(vec_cp_kalman_avg[0]<0) {
                        std::cout<<endl<<"Left ";
                        simulate_key_press(LEFT_KEY);
                    }

        if(vec_cp_kalman_avg[1]<0) {
                        std::cout<<endl<<"Up ";
                        simulate_key_press(UP_KEY);
                    }
        else if(vec_cp_kalman_avg[1]>0) {
                        std::cout<<endl<<"Down ";
                        simulate_key_press(DOWN_KEY);
                    }
	

        g_normal = vec_cp_kalman_l;
        //constructRot(vec_ce_kalman_l[0], vec_ce_kalman_l[1], vec_ce_kalman_l[2]);
        glutPostRedisplay();

        draw_eye_gaze(pt_p_kalman_l, vec_cp_kalman_avg, rect1, frame_clr, 25);				
        draw_eye_gaze(pt_p_kalman_r, vec_cp_kalman_avg, rect2, frame_clr, 25);
        draw_facial_normal(frame_clr, shape, vec_ce_kalman_l, 5*mag_nor);
    }
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
}

static int useRGB = 1;
static int useLighting = 1;
static int useFog = 0;
static int useDB = 1;
static int useLogo = 0;
static int useQuads = 1;

static int tick = -1;
static int moving = 1;

#define GREY    0
#define RED 1
#define GREEN   2
#define BLUE    3
#define CYAN    4
#define MAGENTA 5
#define YELLOW  6
#define BLACK   7

static float materialColor[8][4] =
{
  {0.8, 0.8, 0.8, 1.0},
  {0.8, 0.0, 0.0, 1.0},
  {0.0, 0.8, 0.0, 1.0},
  {0.0, 0.0, 0.8, 1.0},
  {0.0, 0.8, 0.8, 1.0},
  {0.8, 0.0, 0.8, 1.0},
  {0.8, 0.8, 0.0, 1.0},
  {0.0, 0.0, 0.0, 0.6},
};

static float lightPos[4] =
{2.0, 4.0, 2.0, 1.0};
#if 0
static float lightDir[4] =
{-2.0, -4.0, -2.0, 1.0};
#endif
static float lightAmb[4] =
{0.2, 0.2, 0.2, 1.0};
static float lightDiff[4] =
{0.8, 0.8, 0.8, 1.0};
static float lightSpec[4] =
{0.4, 0.4, 0.4, 1.0};

static float groundPlane[4] =
{0.0, 1.0, 0.0, 1.499};
static float backPlane[4] =
{0.0, 0.0, 1.0, 0.899};

static float fogColor[4] =
{0.0, 0.0, 0.0, 0.0};
static float fogIndex[1] =
{0.0};

static unsigned char shadowPattern[128] =
{
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,  /* 50% Grey */
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55,
  0xaa, 0xaa, 0xaa, 0xaa, 0x55, 0x55, 0x55, 0x55
};

static unsigned char sgiPattern[128] =
{
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  /* SGI Logo */
  0xff, 0xbd, 0xff, 0x83, 0xff, 0x5a, 0xff, 0xef,
  0xfe, 0xdb, 0x7f, 0xef, 0xfd, 0xdb, 0xbf, 0xef,
  0xfb, 0xdb, 0xdf, 0xef, 0xf7, 0xdb, 0xef, 0xef,
  0xfb, 0xdb, 0xdf, 0xef, 0xfd, 0xdb, 0xbf, 0x83,
  0xce, 0xdb, 0x73, 0xff, 0xb7, 0x5a, 0xed, 0xff,
  0xbb, 0xdb, 0xdd, 0xc7, 0xbd, 0xdb, 0xbd, 0xbb,
  0xbe, 0xbd, 0x7d, 0xbb, 0xbf, 0x7e, 0xfd, 0xb3,
  0xbe, 0xe7, 0x7d, 0xbf, 0xbd, 0xdb, 0xbd, 0xbf,
  0xbb, 0xbd, 0xdd, 0xbb, 0xb7, 0x7e, 0xed, 0xc7,
  0xce, 0xdb, 0x73, 0xff, 0xfd, 0xdb, 0xbf, 0xff,
  0xfb, 0xdb, 0xdf, 0x87, 0xf7, 0xdb, 0xef, 0xfb,
  0xf7, 0xdb, 0xef, 0xfb, 0xfb, 0xdb, 0xdf, 0xfb,
  0xfd, 0xdb, 0xbf, 0xc7, 0xfe, 0xdb, 0x7f, 0xbf,
  0xff, 0x5a, 0xff, 0xbf, 0xff, 0xbd, 0xff, 0xc3,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

static float cube_vertexes[6][4][4] =
{
  {
    {-1.0, -1.0, -1.0, 1.0},
    {-1.0, -1.0, 1.0, 1.0},
    {-1.0, 1.0, 1.0, 1.0},
    {-1.0, 1.0, -1.0, 1.0}},

    {
        {1.0, 1.0, 1.0, 1.0},
        {1.0, -1.0, 1.0, 1.0},
        {1.0, -1.0, -1.0, 1.0},
        {1.0, 1.0, -1.0, 1.0}},

        {
            {-1.0, -1.0, -1.0, 1.0},
            {1.0, -1.0, -1.0, 1.0},
            {1.0, -1.0, 1.0, 1.0},
            {-1.0, -1.0, 1.0, 1.0}},

            {
                {1.0, 1.0, 1.0, 1.0},
                {1.0, 1.0, -1.0, 1.0},
                {-1.0, 1.0, -1.0, 1.0},
                {-1.0, 1.0, 1.0, 1.0}},

                {
                    {-1.0, -1.0, -1.0, 1.0},
                    {-1.0, 1.0, -1.0, 1.0},
                    {1.0, 1.0, -1.0, 1.0},
                    {1.0, -1.0, -1.0, 1.0}},

                    {
                        {1.0, 1.0, 1.0, 1.0},
                        {-1.0, 1.0, 1.0, 1.0},
                        {-1.0, -1.0, 1.0, 1.0},
                        {1.0, -1.0, 1.0, 1.0}}
                    };

                    static float cube_normals[6][4] =
                    {
                      {-1.0, 0.0, 0.0, 0.0},
                      {1.0, 0.0, 0.0, 0.0},
                      {0.0, -1.0, 0.0, 0.0},
                      {0.0, 1.0, 0.0, 0.0},
                      {0.0, 0.0, -1.0, 0.0},
                      {0.0, 0.0, 1.0, 0.0}
                  };

                  static void
                  usage(void)
                  {
                      printf("\n");
                      printf("usage: scube [options]\n");
                      printf("\n");
                      printf("    display a spinning cube and its shadow\n");
                      printf("\n");
                      printf("  Options:\n");
                      printf("    -geometry  window size and location\n");
                      printf("    -c         toggle color index mode\n");
                      printf("    -l         toggle lighting\n");
                      printf("    -f         toggle fog\n");
                      printf("    -db        toggle double buffering\n");
                      printf("    -logo      toggle sgi logo for the shadow pattern\n");
                      printf("    -quads     toggle use of GL_QUADS to draw the checkerboard\n");
                      printf("\n");
#ifndef EXIT_FAILURE    /* should be defined by ANSI C
                           <stdlib.h> */
#define EXIT_FAILURE 1
#endif
                      exit(EXIT_FAILURE);
                  }

                  void
                  buildColormap(void)
                  {
                      if (useRGB) {
                        return;
                    } else {
                        int mapSize = 1 << glutGet(GLUT_WINDOW_BUFFER_SIZE);
                        int rampSize = mapSize / 8;
                        int entry;
                        int i;

                        for (entry = 0; entry < mapSize; ++entry) {
                          int hue = entry / rampSize;
                          GLfloat val = (entry % rampSize) * (1.0 / (rampSize - 1));
                          GLfloat red, green, blue;

                          red = (hue == 0 || hue == 1 || hue == 5 || hue == 6) ? val : 0;
                          green = (hue == 0 || hue == 2 || hue == 4 || hue == 6) ? val : 0;
                          blue = (hue == 0 || hue == 3 || hue == 4 || hue == 5) ? val : 0;

                          glutSetColor(entry, red, green, blue);
                      }

                      for (i = 0; i < 8; ++i) {
                          materialColor[i][0] = i * rampSize + 0.2 * (rampSize - 1);
                          materialColor[i][1] = i * rampSize + 0.8 * (rampSize - 1);
                          materialColor[i][2] = i * rampSize + 1.0 * (rampSize - 1);
                          materialColor[i][3] = 0.0;
                      }

                      fogIndex[0] = -0.2 * (rampSize - 1);
                  }
              }

              static void
              setColor(int c)
              {
                  if (useLighting) {
                    if (useRGB) {
                      glMaterialfv(GL_FRONT_AND_BACK,
                        GL_AMBIENT_AND_DIFFUSE, &materialColor[c][0]);
                  } else {
                      glMaterialfv(GL_FRONT_AND_BACK,
                        GL_COLOR_INDEXES, &materialColor[c][0]);
                  }
              } else {
                if (useRGB) {
                  glColor4fv(&materialColor[c][0]);
              } else {
                  glIndexf(materialColor[c][1]);
              }
          }
      }

      static void
      drawCube(int color)
      {
          int i;

          setColor(color);

          for (i = 0; i < 6; ++i) {
            glNormal3fv(&cube_normals[i][0]);
            glBegin(GL_POLYGON);
            glVertex4fv(&cube_vertexes[i][0][0]);
            glVertex4fv(&cube_vertexes[i][1][0]);
            glVertex4fv(&cube_vertexes[i][2][0]);
            glVertex4fv(&cube_vertexes[i][3][0]);
            glEnd();
        }
    }

    static void
    drawCheck(int w, int h, int evenColor, int oddColor)
    {
      static int initialized = 0;
      static int usedLighting = 0;
      static GLuint checklist = 0;

      if (!initialized || (usedLighting != useLighting)) {
        static float square_normal[4] =
        {0.0, 0.0, 1.0, 0.0};
        static float square[4][4];
        int i, j;

        if (!checklist) {
          checklist = glGenLists(1);
      }
      glNewList(checklist, GL_COMPILE_AND_EXECUTE);

      if (useQuads) {
          glNormal3fv(square_normal);
          glBegin(GL_QUADS);
      }
      for (j = 0; j < h; ++j) {
          for (i = 0; i < w; ++i) {
            square[0][0] = -1.0 + 2.0 / w * i;
            square[0][1] = -1.0 + 2.0 / h * (j + 1);
            square[0][2] = 0.0;
            square[0][3] = 1.0;

            square[1][0] = -1.0 + 2.0 / w * i;
            square[1][1] = -1.0 + 2.0 / h * j;
            square[1][2] = 0.0;
            square[1][3] = 1.0;

            square[2][0] = -1.0 + 2.0 / w * (i + 1);
            square[2][1] = -1.0 + 2.0 / h * j;
            square[2][2] = 0.0;
            square[2][3] = 1.0;

            square[3][0] = -1.0 + 2.0 / w * (i + 1);
            square[3][1] = -1.0 + 2.0 / h * (j + 1);
            square[3][2] = 0.0;
            square[3][3] = 1.0;

            if (i & 1 ^ j & 1) {
              setColor(oddColor);
          } else {
              setColor(evenColor);
          }

          if (!useQuads) {
              glBegin(GL_POLYGON);
          }
          glVertex4fv(&square[0][0]);
          glVertex4fv(&square[1][0]);
          glVertex4fv(&square[2][0]);
          glVertex4fv(&square[3][0]);
          if (!useQuads) {
              glEnd();
          }
      }
  }

  if (useQuads) {
      glEnd();
  }
  glEndList();

  initialized = 1;
  usedLighting = useLighting;
} else {
    glCallList(checklist);
}
}

static void
myShadowMatrix(float ground[4], float light[4])
{
  float dot;
  float shadowMat[4][4];

  dot = ground[0] * light[0] +
  ground[1] * light[1] +
  ground[2] * light[2] +
  ground[3] * light[3];

  shadowMat[0][0] = dot - light[0] * ground[0];
  shadowMat[1][0] = 0.0 - light[0] * ground[1];
  shadowMat[2][0] = 0.0 - light[0] * ground[2];
  shadowMat[3][0] = 0.0 - light[0] * ground[3];

  shadowMat[0][1] = 0.0 - light[1] * ground[0];
  shadowMat[1][1] = dot - light[1] * ground[1];
  shadowMat[2][1] = 0.0 - light[1] * ground[2];
  shadowMat[3][1] = 0.0 - light[1] * ground[3];

  shadowMat[0][2] = 0.0 - light[2] * ground[0];
  shadowMat[1][2] = 0.0 - light[2] * ground[1];
  shadowMat[2][2] = dot - light[2] * ground[2];
  shadowMat[3][2] = 0.0 - light[2] * ground[3];

  shadowMat[0][3] = 0.0 - light[3] * ground[0];
  shadowMat[1][3] = 0.0 - light[3] * ground[1];
  shadowMat[2][3] = 0.0 - light[3] * ground[2];
  shadowMat[3][3] = dot - light[3] * ground[3];

  glMultMatrixf((const GLfloat *) shadowMat);
}

static char *windowNameRGBDB = "shadow cube (OpenGL RGB DB)";
static char *windowNameRGB = "shadow cube (OpenGL RGB)";
static char *windowNameIndexDB = "shadow cube (OpenGL Index DB)";
static char *windowNameIndex = "shadow cube (OpenGL Index)";

void
idle(void)
{
  tick++;
  if (tick >= 120) {
    tick = 0;
}
glutPostRedisplay();
}

/* ARGSUSED1 */
void
keyboard(unsigned char ch, int x, int y)
{
  switch (ch) {
  case 27:             /* escape */
    exit(0);
    break;
    case 'L':
    case 'l':
    useLighting = !useLighting;
    useLighting ? glEnable(GL_LIGHTING) :
    glDisable(GL_LIGHTING);
    glutPostRedisplay();
    break;
    case 'F':
    case 'f':
    useFog = !useFog;
    useFog ? glEnable(GL_FOG) : glDisable(GL_FOG);
    glutPostRedisplay();
    break;
    case '1':
    glFogf(GL_FOG_MODE, GL_LINEAR);
    glutPostRedisplay();
    break;
    case '2':
    glFogf(GL_FOG_MODE, GL_EXP);
    glutPostRedisplay();
    break;
    case '3':
    glFogf(GL_FOG_MODE, GL_EXP2);
    glutPostRedisplay();
    break;
    case ' ':
    if (!moving) {
      idle();
      glutPostRedisplay();
  }
}
}

void
display(void)
{
  GLfloat cubeXform[4][4];

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glPushMatrix();
  glTranslatef(0.0, -1.5, 0.0);
  glRotatef(-90.0, 1, 0, 0);
  glScalef(2.0, 2.0, 2.0);

  drawCheck(6, 6, BLUE, YELLOW);  /* draw ground */
  glPopMatrix();

  glPushMatrix();
  glTranslatef(0.0, 0.0, -0.9);
  glScalef(2.0, 2.0, 2.0);

  drawCheck(6, 6, BLUE, YELLOW);  /* draw back */
  glPopMatrix();

  glPushMatrix();
  glTranslatef(0.0, 0.2, 0.0);
  glScalef(0.3, 0.3, 0.3);
  /*glRotatef(180*acos(g_normal[1])/(3.14), 1, 0, 0);
  glRotatef(180*acos(-g_normal[0])/(3.14), 0, 1, 0);
  glRotatef(180*acos(g_normal[2])/(3.14), 0, 0, 1);
  */
  glRotatef(180*fabs(acos(-g_normal[2]))/3.14, g_normal[0], g_normal[1], 0.0);
  glScalef(1.0, 1.0, 1.0);
  glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat *) cubeXform);

  drawCube(RED);        /* draw cube */
  glPopMatrix();

  glDepthMask(GL_FALSE);
  if (useRGB) {
    glEnable(GL_BLEND);
} else {
    glEnable(GL_POLYGON_STIPPLE);
}
if (useFog) {
    glDisable(GL_FOG);
}
glPushMatrix();
myShadowMatrix(groundPlane, lightPos);
glTranslatef(0.0, 0.0, 2.0);
glMultMatrixf((const GLfloat *) cubeXform);

  drawCube(BLACK);      /* draw ground shadow */
glPopMatrix();

glPushMatrix();
myShadowMatrix(backPlane, lightPos);
glTranslatef(0.0, 0.0, 2.0);
glMultMatrixf((const GLfloat *) cubeXform);

  drawCube(BLACK);      /* draw back shadow */
glPopMatrix();

glDepthMask(GL_TRUE);
if (useRGB) {
    glDisable(GL_BLEND);
} else {
    glDisable(GL_POLYGON_STIPPLE);
}
if (useFog) {
    glEnable(GL_FOG);
}
if (useDB) {
    glutSwapBuffers();
} else {
    glFlush();
}
}

void
fog_select(int fog)
{
  glFogf(GL_FOG_MODE, fog);
  glutPostRedisplay();
}

void
menu_select(int mode)
{
  switch (mode) {
      case 1:
      moving = 1;
      glutIdleFunc(idle);
      break;
      case 2:
      moving = 0;
      glutIdleFunc(NULL);
      break;
      case 3:
      useFog = !useFog;
      useFog ? glEnable(GL_FOG) : glDisable(GL_FOG);
      glutPostRedisplay();
      break;
      case 4:
      useLighting = !useLighting;
      useLighting ? glEnable(GL_LIGHTING) :
      glDisable(GL_LIGHTING);
      glutPostRedisplay();
      break;
      case 5:
      exit(0);
      break;
  }
}

void
visible(int state)
{
  if (state == GLUT_VISIBLE) {
    if (moving)
      glutIdleFunc(idle);
} else {
    if (moving)
      glutIdleFunc(NULL);
}
}


int main(int argc, char** argv) {

    pthread_t tId;
    pthread_attr_t tAttr;
    pthread_attr_init(&tAttr);
    pthread_create(&tId, &tAttr, startOCV, NULL);

    int width = 350, height = 350;
    int i;
    char *name;
    int fog_menu;

    glutInitWindowSize(width, height);
    glutInit(&argc, argv);
    for (i = 1; i < argc; ++i) {
        if (!strcmp("-c", argv[i])) {
          useRGB = !useRGB;
      } else if (!strcmp("-l", argv[i])) {
          useLighting = !useLighting;
      } else if (!strcmp("-f", argv[i])) {
          useFog = !useFog;
      } else if (!strcmp("-db", argv[i])) {
          useDB = !useDB;
      } else if (!strcmp("-logo", argv[i])) {
          useLogo = !useLogo;
      } else if (!strcmp("-quads", argv[i])) {
          useQuads = !useQuads;
      } else {
          usage();
      }
  }

  /* choose visual */
  if (useRGB) {
    if (useDB) {
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
      name = windowNameRGBDB;
  } else {
      glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
      name = windowNameRGB;
  }
} else {
    if (useDB) {
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_INDEX | GLUT_DEPTH);
      name = windowNameIndexDB;
  } else {
      glutInitDisplayMode(GLUT_SINGLE | GLUT_INDEX | GLUT_DEPTH);
      name = windowNameIndex;
  }
}

glutCreateWindow(name);

buildColormap();

glutKeyboardFunc(keyboard);
glutDisplayFunc(display);
glutVisibilityFunc(visible);

fog_menu = glutCreateMenu(fog_select);
glutAddMenuEntry("Linear fog", GL_LINEAR);
glutAddMenuEntry("Exp fog", GL_EXP);
glutAddMenuEntry("Exp^2 fog", GL_EXP2);

glutCreateMenu(menu_select);
glutAddMenuEntry("Start motion", 1);
glutAddMenuEntry("Stop motion", 2);
glutAddMenuEntry("Toggle fog", 3);
glutAddMenuEntry("Toggle lighting", 4);
glutAddSubMenu("Fog type", fog_menu);
glutAddMenuEntry("Quit", 5);
glutAttachMenu(GLUT_RIGHT_BUTTON);

  /* setup context */
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
glFrustum(-1.0, 1.0, -1.0, 1.0, 1.0, 3.0);

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
glTranslatef(0.0, 0.0, -2.0);

glEnable(GL_DEPTH_TEST);

if (useLighting) {
    glEnable(GL_LIGHTING);
}
glEnable(GL_LIGHT0);
glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpec);

glEnable(GL_NORMALIZE);

if (useFog) {
    glEnable(GL_FOG);
}
glFogfv(GL_FOG_COLOR, fogColor);
glFogfv(GL_FOG_INDEX, fogIndex);
glFogf(GL_FOG_MODE, GL_EXP);
glFogf(GL_FOG_DENSITY, 0.5);
glFogf(GL_FOG_START, 1.0);
glFogf(GL_FOG_END, 3.0);

glEnable(GL_CULL_FACE);
glCullFace(GL_BACK);

glShadeModel(GL_SMOOTH);

glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
if (useLogo) {
    glPolygonStipple((const GLubyte *) sgiPattern);
} else {
    glPolygonStipple((const GLubyte *) shadowPattern);
}

glClearColor(0.0, 0.0, 0.0, 1);
glClearIndex(0);
glClearDepth(1);

glutMainLoop();
  return 0;             /* ANSI C requires main to return int. */
}