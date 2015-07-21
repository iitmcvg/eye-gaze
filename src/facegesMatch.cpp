#include <math.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>

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

	std::vector<std::vector<double> > arr1, arr2;

	read_vector_from_file(argv[1], arr1);
	read_vector_from_file(argv[2], arr2);

	std::cout<<"DTW Score : "<<DTWScore(arr1, arr2)<<std::endl;

	return 0;
}