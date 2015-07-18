#ifndef KMEANS_UTILs_H
#define KMEANS_UTILs_H

void kmeans_array_generate(cv::Mat src, std::vector<float >& vec, int mode);
void kmeans_clusters_view(cv::Mat& src, std::vector<int> labels);

#endif