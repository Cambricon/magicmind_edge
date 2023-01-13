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
void PostProcess(int32_t *pth, const std::string output_dir, const std::string name, const magicmind::Dims &input_dim);
#endif //_POST_PROCESS_HPP
