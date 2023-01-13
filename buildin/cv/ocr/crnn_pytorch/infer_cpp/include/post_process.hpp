#ifndef _SAMPLE_POST_PROCESS_HPP
#define _SAMPLE_POST_PROCESS_HPP

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

static void log_softmax_cpu(float *logits, int max_len, int dict_len);
static void softmax_cpu(float *logits, int max_len, int dict_len);
static string labels2string(vector<int> &label);
static string B_operation(vector<int> &labels, int blank);
static void argmax(float *emission_log_prob, vector<int> &labels, int max_len, int dict_len);
static string greedy_decode(float *emission_log_prob, int max_len, int dict_len);
string post_process(float *logits);

#endif
