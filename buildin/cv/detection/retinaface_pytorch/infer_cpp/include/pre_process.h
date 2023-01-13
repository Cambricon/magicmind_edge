#ifndef _PRE_PROCESS_HPP
#define _PRE_PROCESS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <mm_runtime.h>

// static
std::vector<std::string> LoadImages(const std::string image_dir, int batch_size);

std::pair<cv::Mat, float> Preprocess(cv::Mat img, const magicmind::Dims &input_dim);

#endif //_PRE_PROCESS_HPP
