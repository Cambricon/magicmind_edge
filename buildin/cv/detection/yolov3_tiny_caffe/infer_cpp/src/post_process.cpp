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

float cal_dimension(float detection, float length, float scale, float input_dim)
{
    float real_detection = (detection * input_dim - (input_dim - length * scale) / 2) / scale;
    real_detection = std::max(float(0.0), real_detection);
    real_detection = std::min(length, real_detection);
    return real_detection;
}

std::vector<BBox> PostProcess(cv::Mat &img, std::vector<std::vector<float>> results, std::map<int, std::string> imagenet_name_map, const std::string name, const std::string output_dir, bool save_img)
{
    // std::string filename = output_dir + "/" + name + ".txt";
    // std::ofstream file_map(filename);
    int src_h = img.rows;
    int src_w = img.cols;
    int dst_h = 416;
    int dst_w = 416;
    float ratio = std::min(float(dst_h)/float(src_h), float(dst_w)/float(src_w));
    std::vector<BBox> bboxes;
    int detect_num = results.size();
    for (int i = 0; i < detect_num; ++i)
    {
        int detect_class = results[i][0];
        float score = results[i][1];
        float xmin = results[i][2];
        float ymin = results[i][3];
        float xmax = results[i][4];
        float ymax = results[i][5];

        xmin = cal_dimension(xmin, src_w, ratio, dst_w);
        ymin = cal_dimension(ymin, src_h, ratio, dst_h);
        xmax = cal_dimension(xmax, src_w, ratio, dst_w);
        ymax = cal_dimension(ymax, src_h, ratio, dst_h);

        BBox bbox = {
            detect_class,  // category id
            score,  // score
            xmin,  // left
            ymin,   // top
            xmax,  // right
            ymax,   // bottom
        };
        bboxes.emplace_back(bbox);
        // file_map << imagenet_name_map[detect_class] << ","
        //          << score << ","
        //          << xmin << ","
        //          << ymin << ","
        //          << xmax << ","
        //          << ymax << "\n";
        cv::rectangle(img, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 255, 0));
        auto fontface = cv::FONT_HERSHEY_TRIPLEX;
        double fontscale = 0.5;
        int thickness = 1;
        int baseline = 0;
        std::string text = imagenet_name_map[detect_class] + ": " + std::to_string(score);
        cv::Size text_size = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
        cv::putText(img, text, cv::Point(xmin, ymin + text_size.height), fontface, fontscale, cv::Scalar(255, 255, 255), thickness);
    }
    if (save_img)
    {
        imwrite(output_dir + "/" + name + ".jpg", img);
    }
    // file_map.close();
    return bboxes;
}
