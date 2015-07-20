#include <math.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/gui_widgets.h"

#include "faceDetection.h"
#include "util.h"
#include "gestureDetection.h"

int main(int argc, char **argv) {

	cv::VideoCapture cap(0);

	FaceFeatures *face_features = new FaceFeatures();
	FaceData *face_data = new FaceData();
	FacePose *face_pose = new FacePose();
	FaceGesture *face_gesture = new FaceGesture();

	face_gesture->assign(15);

	cv::Mat frame, frame_clr;

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;

	dlib::deserialize("./res/shape_predictor_68_face_landmarks.dat") >> pose_model;

	std::vector<std::vector<std::vector<double> > > gestures_learned(2);

	read_vector_from_file(argv[1], gestures_learned[0]);
	read_vector_from_file(argv[2], gestures_learned[1]);

	int score_temp_1, score_temp_2, pos;
	std::vector<double> vec_normal(3);
	std::vector<std::vector<double> > vec_current_bin(0);

	while(1) {
		try {
			cap>>frame;
			cv::flip(frame, frame, 1);
			frame.copyTo(frame_clr);
			cv::cvtColor(frame, frame, CV_BGR2GRAY);

			dlib::cv_image<unsigned char> cimg_gray(frame);
			dlib::cv_image<dlib::bgr_pixel> cimg_clr(frame_clr);

			std::vector<dlib::rectangle> faces = detector(cimg_gray);

			std::vector<dlib::full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg_gray, faces[i]));

			if(faces.size()) {

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

				std::cout<<face_pose->normal[0]<<" "<<face_pose->normal[1]<<" "<<face_pose->normal[2]<<"\n";

				vec_normal[0] = face_pose->normal[0];
				vec_normal[1] = face_pose->normal[1];
				vec_normal[2] = face_pose->normal[2];

				face_gesture->normal->push(vec_normal);
				//std::cout<<"Filled : "<<face_gesture->normal->get_filled()<<std::endl;

				vec_current_bin = face_gesture->normal->clone();
				//std::cout<<"Size : "<<vec_current_bin.size()<<std::endl;

				for(int i=0;i<vec_current_bin.size();i++) {
					std::cout<<vec_current_bin[i][0]<<std::endl;
				}

				score_temp_1 = DTWScore(vec_current_bin, gestures_learned[0]);
				pos = 0;

				for(int i=1;i<gestures_learned.size();i++) {
					score_temp_2 = DTWScore(vec_current_bin, gestures_learned[i]);
					if(score_temp_2 < score_temp_1) {
						score_temp_1 = score_temp_2;
						pos = i;
					}
				}

				std::cout<<"Matched with gesture["<<pos<<"]"<<std::endl;
			}
			else {
				std::cout<<"Zero faces"<<std::endl;
			}
		}
		catch(std::exception& e) {
			std::cout<<e.what()<<std::endl;
			break;
		}
	}
	return 0;
}