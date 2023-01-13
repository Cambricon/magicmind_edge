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

static float contourScore(const Mat& pred, const vector<Point>& contour);
static void unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly, float unclipRatio);
void post_process(Mat pred, vector<vector<Point2f>> &boxes, vector<float> &confidences, int width, int height, float threshold_);

void drawboxes(vector<vector<Point2f>> boxes, Mat img_show, const std::string output_dir, const std::string name);
#endif
