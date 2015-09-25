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

#include "util.h"
#include "constants.h"
#include "faceDetection.h"

using namespace std;
using namespace dlib;

void faceModel::assign(full_object_detection shape) {
	faceShape = shape;
}