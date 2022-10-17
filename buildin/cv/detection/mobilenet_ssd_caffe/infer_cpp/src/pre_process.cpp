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
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(dst_w, dst_h));
    cv::Mat mat_fp32(dst_h, dst_w, CV_32FC3);
    resized.convertTo(mat_fp32, CV_32FC3, 1 / 127.5f, -1);
    return mat_fp32;
}
