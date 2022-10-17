#include <glog/logging.h>
#include "post_process.h"
#include "utils.h"

static std::vector<std::string> glabels = {
    "__background__",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
};

std::vector<std::vector<BBox>> PostProcess(cv::Mat &img, int batch_size, const magicmind::Dims &preds_dim, float conf, float* preds, const std::string name, const std::string output_dir, bool save_img)
{
    static constexpr int bbox_size = 7;  // every 7 values form a bounding box.
    std::vector<std::vector<BBox>> bboxes;
    bboxes.resize(batch_size);
    int bbox_num = static_cast<int>(preds_dim[0]);
    for (int i = 0; i < bbox_num; ++i) {
        float *bbox_data = preds + i * bbox_size;
        int batch_id = static_cast<int>(bbox_data[0]);
        int category = static_cast<int>(bbox_data[1]);
        if (0 == category) {
          // background
          continue;
        }
        if (conf > bbox_data[2]) continue;
        BBox bbox = {
          category,
          bbox_data[2],  // score
          static_cast<int>(std::floor(bbox_data[3] * img.cols)),  // left
          static_cast<int>(std::floor(bbox_data[4] * img.rows)),   // top
          static_cast<int>(std::floor(bbox_data[5] * img.cols)),  // right
          static_cast<int>(std::floor(bbox_data[6] * img.rows))    // bottom
        };
        if (bbox.left >= bbox.right || bbox.top >= bbox.bottom) continue;
        // check border
        bbox.left = bbox.left > 0 ? bbox.left : 0;
        bbox.right = bbox.right < img.cols ? bbox.right : img.cols;
        bbox.top = bbox.top > 0 ? bbox.top : 0;
        bbox.bottom = bbox.bottom < img.rows ? bbox.bottom : img.rows;
        bboxes[batch_id].emplace_back(bbox);
        if (save_img) {
            cv::rectangle(img, cv::Rect(cv::Point(bbox.left, bbox.top), cv::Point(bbox.right, bbox.bottom)), cv::Scalar(0, 255, 0));
            auto fontface = cv::FONT_HERSHEY_TRIPLEX;
            double fontscale = 0.5;
            int thickness = 1;
            int baseline = 0;
            std::string text = (bbox.label < glabels.size() ? glabels[bbox.label] : "Unknow label") + ":" +std::to_string(bbox.score);
            cv::Size text_size = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
            cv::putText(img, text, cv::Point(bbox.left, bbox.top + text_size.height), fontface, fontscale, cv::Scalar(255, 255, 255), thickness);
            imwrite(output_dir + "/" + name + ".jpg", img);
       }
    }
    return bboxes;
}
