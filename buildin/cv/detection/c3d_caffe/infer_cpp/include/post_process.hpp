#ifndef _C3D_POST_PROCESS_HPP
#define _C3D_POST_PROCESS_HPP

#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include <deque>
using namespace std;
std::vector<int> sort_indexes(const std::vector<float> &v, bool reverse = false);
#endif //_C3D_POST_PROCESS_HPP