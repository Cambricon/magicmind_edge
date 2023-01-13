#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cmath>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

#include "post_process.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

static float contourScore(const Mat& pred, const vector<Point>& contour)
{
    Rect rect = boundingRect(contour);
    int xmin = max(rect.x, 0);
    int xmax = min(rect.x + rect.width, pred.cols - 1);
    int ymin = max(rect.y, 0);
    int ymax = min(rect.y + rect.height, pred.rows - 1);

    Mat binROI = pred(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

    Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    vector<Point> roiContour;
    for (size_t i = 0; i < contour.size(); i++) {
        Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
        roiContour.push_back(pt);
    }
    vector<vector<Point>> roiContours = {roiContour};
    fillPoly(mask, roiContours, Scalar(1));
    float score = mean(binROI, mask).val[0];
    return score;
}

static void unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly, float unclipRatio)
{
    float area = contourArea(inPoly);
    float length = arcLength(inPoly, true);
    float distance = area * unclipRatio / length;

    size_t numPoints = inPoly.size();
    vector<vector<Point2f>> newLines;
    for (size_t i = 0; i < numPoints; i++)
    {
        vector<Point2f> newLine;
        Point pt1 = inPoly[i];
        Point pt2 = inPoly[(i - 1) % numPoints];
        Point vec = pt1 - pt2;
        float unclipDis = (float)(distance / norm(vec));
        Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
        newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
        newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
        newLines.push_back(newLine);
    }

    size_t numLines = newLines.size();
    for (size_t i = 0; i < numLines; i++)
    {
        Point2f a = newLines[i][0];
        Point2f b = newLines[i][1];
        Point2f c = newLines[(i + 1) % numLines][0];
        Point2f d = newLines[(i + 1) % numLines][1];
        Point2f pt;
        Point2f v1 = b - a;
        Point2f v2 = d - c;
        float cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

        if( fabs(cosAngle) > 0.7 )
        {
            pt.x = (b.x + c.x) * 0.5;
            pt.y = (b.y + c.y) * 0.5;
        }
        else
        {
            float denom = a.x * (float)(d.y - c.y) + b.x * (float)(c.y - d.y) +
                          d.x * (float)(b.y - a.y) + c.x * (float)(a.y - b.y);
            float num = a.x * (float)(d.y - c.y) + c.x * (float)(a.y - d.y) + d.x * (float)(c.y - a.y);
            float s = num / denom;

            pt.x = a.x + s*(b.x - a.x);
            pt.y = a.y + s*(b.y - a.y);
        }
        outPoly.push_back(pt);
    }
}

void post_process(Mat pred, vector<vector<Point2f>> &boxes, vector<float> &confidences, int width, int height, float threshold_)
{    
    // Threshold
    Mat bitmap;
    threshold(pred, bitmap, threshold_, 255, THRESH_BINARY);
    // Scale ratio
    float scaleHeight = (float)(height) / (float)(pred.size[0]);
    float scaleWidth = (float)(width) / (float)(pred.size[1]);
    // Find contours
    vector< vector<Point> > contours;
    bitmap.convertTo(bitmap, CV_8U);
    imwrite("bitmap.jpg", bitmap);
    findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Candidate number limitation
    size_t numCandidate = min((int)contours.size(), 100);
    for (size_t i = 0; i < numCandidate; i++)
    {
        vector<Point>& contour = contours[i];

        // Calculate text contour score
        confidences.emplace_back(contourScore(pred, contour));
        // if (contourScore(pred, contour) < polygonThreshold)
        //     continue;

        // Rescale
        vector<Point> contourScaled; contourScaled.reserve(contour.size());
        for (size_t j = 0; j < contour.size(); j++)
        {
            contourScaled.push_back(Point(int(contour[j].x * scaleWidth),
                                          int(contour[j].y * scaleHeight)));
        }

        // Unclip
        RotatedRect bounding_box = minAreaRect(contourScaled);

        // minArea() rect is not normalized, it may return rectangles with angle=-90 or height < width
        const float angle_threshold = 60;  // do not expect vertical text, TODO detection algo property
        bool swap_size = false;
        if (bounding_box.size.width < bounding_box.size.height)  // horizontal-wide text area is expected
            swap_size = true;
        else if (fabs(bounding_box.angle) >= angle_threshold)  // don't work with vertical rectangles
            swap_size = true;
        if (swap_size)
        {
            swap(bounding_box.size.width, bounding_box.size.height);
            if (bounding_box.angle < 0)
                bounding_box.angle += 90;
            else if (bounding_box.angle > 0)
                bounding_box.angle -= 90;
        }

        Point2f vertex[4];
        bounding_box.points(vertex);  // order: bl, tl, tr, br
        vector<Point2f> approx;
        for (int j = 0; j < 4; j++)
            approx.emplace_back(vertex[j]);
        vector<Point2f> polygon;
        unclip(approx, polygon, 1.5);
        boxes.push_back(polygon);
    }
}

void drawboxes(vector<vector<Point2f>> boxes, Mat img_show, const std::string output_dir, const std::string name)
{    
    for(int i=0; i<boxes.size(); ++i)
    {        
        line(img_show, Point((int)boxes[i][0].x, (int)boxes[i][0].y), Point((int)boxes[i][1].x, (int)boxes[i][1].y), Scalar(0, 255, 0), 1);
        line(img_show, Point((int)boxes[i][1].x, (int)boxes[i][1].y), Point((int)boxes[i][2].x, (int)boxes[i][2].y), Scalar(0, 255, 0), 1);
        line(img_show, Point((int)boxes[i][2].x, (int)boxes[i][2].y), Point((int)boxes[i][3].x, (int)boxes[i][3].y), Scalar(0, 255, 0), 1);
        line(img_show, Point((int)boxes[i][3].x, (int)boxes[i][3].y), Point((int)boxes[i][0].x, (int)boxes[i][0].y), Scalar(0, 255, 0), 1);
    }
    
    imwrite(output_dir + "/" + name + ".jpg", img_show);
    // imwrite("img_show_boxes.jpg", img_show);
}
