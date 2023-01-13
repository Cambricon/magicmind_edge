#ifndef _SAMPLE_PRE_PROCESS_HPP
#define _SAMPLE_PRE_PROCESS_HPP
#include <tuple>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <mm_runtime.h>


// cv::Mat Preprocess(cv::Mat img, const magicmind::Dims &input_dim, bool transpose);
cv::Mat Preprocess(cv::Mat img, bool transpose);
std::vector<cv::String> LoadImages(const std::string image_dir, const int batch_size);
#endif

