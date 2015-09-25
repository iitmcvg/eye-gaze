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
    try {
        cv::VideoCapture cap(0);
        image_window win1, win2;

        FaceFeatures *face_features = new FaceFeatures();
        FaceData *face_data = new FaceData();
        FacePose *face_pose = new FacePose();

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;

        vector<double> vec_ce_pos_l(3, 0), vec_ce_pos_r(3, 0);
        vector<double> vec_cp_pos_l(3, 0), vec_cp_pos_r(3, 0);
        vector<double> vec_ep_pos_l(3, 0), vec_ep_pos_r(3, 0);

        cv::Point pt_p_pos_l(0, 0), pt_p_pos_r(0, 0);

        cv::Mat frame, frame_clr, temp1, temp2, temp3, roi_l_clr, roi_r_clr, roi_l_gray, roi_r_gray;
        
        while(!win.is_closed()) {
            cap >> frame_clr;

            cv::flip(frame_clr, frame_clr, 1);
            cv::cvtColor(frame_clr, frame, CV_BGR2GRAY);

            cv_image<unsigned char> cimg_gray(frame);
            cv_image<bgr_pixel> cimg_clr(frame_clr);
            vector<rectangle> faces = detector(cimg_gray);
            vector<full_object_detection> shapes;

            for(unsigned long i = 0; i < (int) faces.size(); ++i)
                faces.push_back(pose_model(cimg_gray, faces[i]));

            if(shapes.size() == 0) cout<<"Zero faces"<<endl;
            else {
                
            }
        }

    }
}
