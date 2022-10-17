#ifndef _PRE_PROCESS_H
#define _PRE_PROCESS_H

#include <map>
#include <regex>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <mm_runtime.h>

cv::Mat Preprocess(cv::Mat img,const magicmind::Dims &input_dim);

std::vector<cv::String> LoadImages(const std::string image_dir, const int batch_size);

#endif //_PRE_PROCESS_H

