#include "pre_process.h"
#include "utils.h"
#include <glog/logging.h>

/**
 * @brief load all images(jpg) from image directory(args.image_dir)
 * @return Returns image paths
 */
std::vector<cv::String> LoadImages(const std::string image_dir, const int batch_size)
{
  char abs_path[PATH_MAX];
  LOG_IF(FATAL, !realpath(image_dir.c_str(), abs_path))
     << "Get real image path failed.";
  std::string glob_path = std::string(abs_path);
  std::vector<cv::String> image_paths;
  cv::glob(glob_path + "/*.jpg", image_paths, false);
  // pad to multiple of batch_size.
  // The program will stuck when the number of input images is not an integer multiple of the batch size
  size_t pad_num = batch_size - image_paths.size() % batch_size;
  if (pad_num != batch_size)
  {
    LOG(INFO) << "There are " << image_paths.size() << " images in total, add " << pad_num
        << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
    while (pad_num--)
    {
      image_paths.emplace_back(*image_paths.rbegin());
    }
  }
  return image_paths;
}

cv::Mat Preprocess(cv::Mat img, int dst_h, int dst_w) {
    // NHWC order implementation. Make sure your model's input is in NHWC order.
    int height = img.rows;
    int width = img.cols;
    int min_side = 320;
    float scale = float(min_side) / float(std::max(height, width));
    int new_w = int(float(width) * scale);
    int new_h = int(float(height) * scale);
    cv::resize(img, img, cv::Size(new_w, new_h));
    int top, bottom, left, right = 0;
    if (new_w % 2 != 0 and new_h % 2 == 0) {
        top = (min_side - new_h) / 2;
        bottom = (min_side - new_h) / 2;
        left = (min_side - new_w) / 2 + 1;
        right = (min_side - new_w) / 2;
    } else if (new_h % 2 != 0 and new_w % 2 == 0) {
        top = (min_side - new_h) / 2 + 1;
        bottom = (min_side - new_h) / 2;
        left = (min_side - new_w) / 2;
        right = (min_side - new_w) / 2;
    } else if (new_h % 2 == 0 and new_w % 2 == 0) {
        top = (min_side - new_h) / 2;
        bottom = (min_side - new_h) / 2;
        left = (min_side - new_w) / 2;
        right = (min_side - new_w) / 2;
    } else {
        top = (min_side - new_h) / 2 + 1;
        bottom = (min_side - new_h) / 2;
        left = (min_side - new_w) / 2 + 1;
        right = (min_side - new_w) / 2;
    }
    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    img.convertTo(img, CV_32F);
    cv::Scalar mean(104, 117, 123);
    img -= mean;

    return img;
}
