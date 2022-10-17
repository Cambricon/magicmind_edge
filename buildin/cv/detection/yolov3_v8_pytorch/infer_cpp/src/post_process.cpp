#include <glog/logging.h>
#include "post_process.h"
#include "utils.h"


std::map<int, std::string> LoadLabelName(std::string name_map_file)
{
    if (!check_file_exist(name_map_file))
    {
        LOG(INFO) << "coco_name file: " + name_map_file + " does not exist.\n";
        exit(0);
    }
    std::map<int, std::string> coco_name_map;
    std::ifstream in(name_map_file);
    if (!in)
    {
        LOG(INFO) << "failed to load coco_name file: " + name_map_file + ".\n";
        exit(0);
    }
    std::string line;
    int index = 0;
    while (getline(in, line))
    {
        coco_name_map[index] = line;
        index += 1;
    }
    return coco_name_map;
}

std::vector<BBox> PostProcess(cv::Mat &img,const int dst_h,const int dst_w, std::vector<std::vector<float>> results, std::map<int, std::string> imagenet_name_map, const std::string name, const std::string output_dir, bool save_img)
{
    int src_h = img.rows;
    int src_w = img.cols;
    std::vector<BBox> bboxes;
    float ratio = std::min(float(dst_h) / float(src_h), float(dst_w) / float(src_w));
    float pad_top = (dst_h - ratio * src_h) / 2;
    float pad_left = (dst_w - ratio * src_w) / 2;
    int detect_num = results.size();
    for (int i = 0; i < detect_num; ++i)
    {
        int detect_class = results[i][0];
        float score = results[i][1];
        float xmin = (results[i][2] * dst_w - pad_left) / ratio;  // left
        float ymin = (results[i][3] * dst_h - pad_top) / ratio;   // top
        float xmax = (results[i][4] * dst_w - pad_left) / ratio;  // right
        float ymax = (results[i][5] * dst_h - pad_top) / ratio;    // bottom
        if (xmin >= xmax || ymin >= ymax) continue;
        xmin = std::min(std::max(float(0.0), xmin), float(dst_w));
        ymin = std::min(std::max(float(0.0), ymin), float(dst_h));
        xmax = std::min(std::max(float(0.0), xmax), float(dst_w));
        ymax = std::min(std::max(float(0.0), ymax), float(dst_h));
        BBox bbox = {
            detect_class,  // category id
            score,  // score
            xmin,  // left
            ymin,   // top
            xmax,  // right
            ymax,   // bottom
        };
        bboxes.emplace_back(bbox);

        if (save_img) {
            cv::rectangle(img, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 255, 0));
            auto fontface = cv::FONT_HERSHEY_TRIPLEX;
            double fontscale = 0.5;
            int thickness = 1;
            int baseline = 0;
            std::string text = imagenet_name_map[detect_class] + ": " + std::to_string(score);
            cv::Size text_size = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
            cv::putText(img, text, cv::Point(xmin, ymin + text_size.height), fontface, fontscale, cv::Scalar(255, 255, 255), thickness);
            imwrite(output_dir + "/" + name + ".jpg", img);
       }
    }
    return bboxes;
}
