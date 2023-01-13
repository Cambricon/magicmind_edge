#ifndef _POST_PROCESS_HPP
#define _POST_PROCESS_HPP

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "utils.h"

std::vector<std::vector<BBox>> PostProcess(cv::Mat &img,int batch_size, const magicmind::Dims &preds_dim, float conf, float * preds, const std::string image_name, const std::string output_dir, bool save_img);
//std::vector<BBox> PostProcess(cv::Mat &img,int dst_h, int dst_w, float conf, float * preds, const std::string image_name, const std::string output_dir, bool save_img);


#endif //_POST_PROCESS_HPP
