#ifndef _PRE_PROCESS_HPP
#define _PRE_PROCESS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat Preprocess(cv::Mat src_img);
std::vector<std::string> LoadImages(const std::string image_dir, const int batch_size, const std::string file_list);
#endif //_PRE_PROCESS_HPP
