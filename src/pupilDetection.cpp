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

using namespace std;
using namespace dlib;

cv::Point unscale_point(cv::Point p, cv::Rect origSize) {
	float ratio = (((float)(50))/origSize.width);
	int x = round(p.x / ratio);
	int y = round(p.y / ratio);
	return cv::Point(x,y);
}

void scale_standard_size(const cv::Mat &src,cv::Mat &dst) {
	cv::resize(src, dst, cv::Size(50,(((float)50)/src.cols) * src.rows));
}

cv::Mat compute_x_gradient(const cv::Mat &mat) {
	cv::Mat out(mat.rows,mat.cols,CV_64F);

	for (int y = 0; y < mat.rows; ++y) {
		const uchar *Mr = mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);

		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < mat.cols - 1; ++x) {
			Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
		}
		Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
	}

	return out;
}

double compute_threshold(const cv::Mat &mat, double stdDevFactor) {
	cv::Scalar stdMagnGrad, meanMagnGrad;
	meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}

cv::Mat get_matrix_magnitude(const cv::Mat &matX, const cv::Mat &matY) {
	cv::Mat mags(matX.rows,matX.cols,CV_64F);
	for (int y = 0; y < matX.rows; ++y) {
		const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < matX.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		}
	}
	return mags;
}

bool check_point_in_mat(cv::Point p,int rows,int cols) {
	return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

bool is_point_in_mat(const cv::Point &np, const cv::Mat &mat) {
	return check_point_in_mat(np, mat.rows, mat.cols);
}

cv::Mat remove_edges(cv::Mat &mat) {
	cv::rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);

	cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
	std::queue<cv::Point> toDo;
	toDo.push(cv::Point(0,0));
	while (!toDo.empty()) {
		cv::Point p = toDo.front();
		toDo.pop();
		if (mat.at<float>(p) == 0.0f) {
			continue;
		}
    // add in every direction
    cv::Point np(p.x + 1, p.y); // right
    if (is_point_in_mat(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (is_point_in_mat(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (is_point_in_mat(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (is_point_in_mat(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
}
return mask;
}

void check_pupil(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
  // for all possible centers
	for (int cy = 0; cy < out.rows; ++cy) {
		double *Or = out.ptr<double>(cy);
		const unsigned char *Wr = weight.ptr<unsigned char>(cy);
		for (int cx = 0; cx < out.cols; ++cx) {
			if (x == cx && y == cy) {
				continue;
			}
      // create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
      // normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx*gx + dy*gy;
			dotProduct = max(0.0,dotProduct);
      // square and multiply by the weight
			if (true) {
				Or[cx] += dotProduct * dotProduct * (Wr[cx]+5);
			} else {
				Or[cx] += dotProduct * dotProduct;
			}
		}
	}
}

cv::Point get_pupil_coordinates(cv::Mat eye_mat,cv::Rect eye) {
	cv::Mat eyeROIUnscaled = eye_mat;
	cv::Mat eyeROI;
	scale_standard_size(eyeROIUnscaled, eyeROI);
  // draw eye region
  //rectangle(face,eye,1234);
  //-- Find the gradient
	cv::Mat gradientX = compute_x_gradient(eyeROI);
	cv::Mat gradientY = compute_x_gradient(eyeROI.t()).t();
  //-- Normalize and threshold the gradient
  // compute all the magnitudes
	cv::Mat mags = get_matrix_magnitude(gradientX, gradientY);
  //compute the threshold
	double gradientThresh = compute_threshold(mags, 50.0);
  //double gradientThresh = kGradientThreshold;
  //double gradientThresh = 0;
  //normalize
	for (int y = 0; y < eyeROI.rows; ++y) {
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < eyeROI.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];
			if (magnitude > gradientThresh) {
				Xr[x] = gX/magnitude;
				Yr[x] = gY/magnitude;
			} else {
				Xr[x] = 0.0;
				Yr[x] = 0.0;
			}
		}
	}

  //imshow(debugWindow,gradientX);

  //-- Create a blurred and inverted image for weighting
	cv::Mat weight;
	GaussianBlur( eyeROI, weight, cv::Size( 5, 5 ), 0, 0 );
	for (int y = 0; y < weight.rows; ++y) {
		unsigned char *row = weight.ptr<unsigned char>(y);
		for (int x = 0; x < weight.cols; ++x) {
			row[x] = (255 - row[x]);
		}
	}

  //-- Run the algorithm!
	cv::Mat outSum = cv::Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
  // for each possible gradient location
  // Note: these loops are reversed from the way the paper does them
  // it evaluates every possible center for each gradient location instead of
  // every possible gradient location for every center.
	//printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);

	for (int y = 0; y < weight.rows; ++y) {
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = 0; x < weight.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0) {
				continue;
			}
			check_pupil(x, y, weight, gX, gY, outSum);
		}
	}
  // scale all the values down, basically averaging them
	double numGradients = (weight.rows*weight.cols);
	cv::Mat out;
	outSum.convertTo(out, CV_32F,1.0/numGradients);
  //imshow(debugWindow,out);
  //-- Find the maximum point
	cv::Point maxP;
	double maxVal;
	minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  //-- Flood fill the edges
	cv::Mat floodClone;
    //double floodThresh = compute_threshold(out, 1.5);
	double floodThresh = maxVal * 0.97;
	threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);

	cv::Mat mask = remove_edges(floodClone);

    // redo max
	minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);


	return unscale_point(maxP, eye);

}
void draw_eye_gaze(cv::Point pt, std::vector<double> vec_gaze, cv::Rect roi_eye, cv::Mat& img) {

	double del_x = 20*vec_gaze[0];
	double del_y = 20*vec_gaze[1];

	cv::line(img, cv::Point(pt.x + roi_eye.x, pt.y + roi_eye.y), cv::Point(pt.x + del_x + roi_eye.x, pt.y + del_y + roi_eye.y), cv::Scalar(255, 255, 255), 1);
}

double get_z_component(double Cf_left, cv::Point pt_p_kalman, cv::Point pt_e_kalman, std::vector<double> vec_ce_kalman) {	
	make_unit_vector(vec_ce_kalman, vec_ce_kalman);
	double x, y, z, mag = Cf_left*13.101, z_comp;
	x = pt_e_kalman.x - vec_ce_kalman[0]*mag;
	y = pt_e_kalman.y - vec_ce_kalman[1]*mag;
	z = -vec_ce_kalman[2]*mag;

	z_comp = sqrt(pow(8*Cf_left, 2) - pow(pt_p_kalman.x - x, 2) + pow(pt_p_kalman.y - y, 2)) - z;
	return z_comp;
}

void retrace_eye_center(cv::Point& pt_e_pos, double normal[3], double mag) {
	pt_e_pos.x = pt_e_pos.x - normal[0]*mag;
	pt_e_pos.y = pt_e_pos.y - normal[1]*mag;
}
