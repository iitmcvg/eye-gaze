#ifndef KMEANS_UTILS_H
#define KMEANS_UTILS_H

void kmeans_array_generate(cv::Mat src, std::vector<float>& vec, int mode);
void kmeans_clusters_view(cv::Mat& src, std::vector<int> labels);

#endif