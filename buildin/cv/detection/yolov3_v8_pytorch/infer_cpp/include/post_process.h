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

std::map<int, std::string> LoadLabelName(std::string name_map_file);

std::vector<BBox> PostProcess(cv::Mat &img,int dst_h, int dst_w, std::vector<std::vector<float>> results, std::map<int, std::string> coco_name_map, const std::string image_name, const std::string output_dir, bool save_img);


#endif //_POST_PROCESS_HPP
