#include "pre_process.h"
#include "utils.h"
#include <glog/logging.h>

/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
std::vector<cv::String> LoadImages(const std::string image_dir, const int batch_size) {
  char abs_path[PATH_MAX];
  LOG(INFO) << image_dir.c_str();
  LOG_IF(FATAL, !realpath(image_dir.c_str(), abs_path)) << "Get real image path failed.";
  std::string glob_path = std::string(abs_path);
  std::vector<cv::String> jpg_paths, jpeg_paths, image_paths;
  cv::glob(glob_path + "/*.jpg", image_paths, false);
  cv::glob(glob_path + "/*.JPEG", image_paths, false);
  if (!jpg_paths.empty())
    image_paths.insert(jpg_paths.begin(), jpg_paths.end(), image_paths.end());
  if (!jpeg_paths.empty())
    image_paths.insert(jpeg_paths.begin(), jpeg_paths.end(), image_paths.end());
  // pad to multiple of batch_size.
  // The program will stuck when the number of input images is not an integer multiple of the batch
  // size
  size_t pad_num = batch_size - image_paths.size() % batch_size;
  if (pad_num != batch_size) {
    LOG(INFO) << "There are " << image_paths.size() << " images in total, add " << pad_num
              << " more images to make the number of images is an integral multiple of batchsize["
              << batch_size << "].";
    while (pad_num--)
      image_paths.emplace_back(*image_paths.rbegin());
  }
  return image_paths;
}

cv::Mat Preprocess(cv::Mat img, const magicmind::Dims &input_dim) {
  // NHWC order implementation. Make sure your model's input is in NHWC order.
  /*
     (x - mean) / std : This calculation process is performed at the first layer of the model,
     See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
  */
  int h = input_dim[1];
  int w = input_dim[2];

  // resize
  float scale = 1.0f * 256 / std::min(img.cols, img.rows);
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(std::round(scale * img.cols), std::round(scale * img.rows)));
  // center crop
  auto roi = resized(cv::Rect((resized.cols - w) / 2, (resized.rows - h) / 2, w, h));
  cv::Mat dst_img(h, w, CV_8UC3);
  roi.copyTo(dst_img);
  return dst_img;
}

