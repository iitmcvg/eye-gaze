#ifndef GESTURE_DETECTION_H
#define GESTURE_DETECTION_H

double maximum(double a, double b, double c);
double measure_deviation(std::vector<double> arr1, std::vector<double> arr2);
double minimum(double a, double b, double c);
double DTWScore(std::vector<std::vector<double> > arr1, std::vector<std::vector<double> > arr2);

struct FixedBin {
	std::vector<std::vector<double> > bin;
	int size;
	int filled;

	void assign(int _size);
	void push(std::vector<double> vec);
	int get_size();
	int get_filled();
	void get(int pos, std::vector<double>& vec);
	std::vector<std::vector<double> > clone();
};

struct FaceGesture {
	FixedBin* normal;

	void assign(int normal_size);
};

#endif