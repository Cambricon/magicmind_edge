#ifndef _POST_PROCESS_H
#define _POST_PROCESS_H

#include <utility>
#include <mm_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <tuple>
#include <utility>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "utils.h"

std::pair<Keypoints, PersonInfos> Postprocess(
      cv::Mat img, float *preds, const float scaling_factor,
      const magicmind::Dims &input_dim, const magicmind::Dims &output_dim);
#endif //_POST_PROCESS_H
