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
#include "../include/utils.h"
#include <mm_runtime.h>

std::vector<BBox> RetinafacePostprocess(
    cv::Mat img, float scaling_factors,
    const magicmind::Dims &input_dim,
    std::vector<float*> outputs,
    std::vector<magicmind::Dims> output_dims, 
    const float confidence_thresholds,
    const float nms_thresholds);

void Draw(cv::Mat img, const std::vector<BBox> &bboxes);

void WritePreds(const std::string img_path, const std::string image_name, 
                const std::vector<BBox> &bboxes);
#endif //_POST_PROCESS_HPP
