#include "pre_process.h"
#include "utils.h"
#include <glog/logging.h>


/**
 * @brief load all images(jpg) from image directory(FLAGS_image_list)
 * @return Returns image paths
 */
// static
std::vector<std::string> LoadImages(const std::string image_dir, int batch_size) {
  std::vector<std::string> image_paths = LoadFileList(image_dir);
  // pad to multiple of batch_size.
  // The program will stuck when the number of input images is not an integer multiple of the batch size
  size_t pad_num = batch_size - image_paths.size() % batch_size;
  if (pad_num != batch_size) {
    LOG(INFO) << "There are " << image_paths.size() << " images in total, add " << pad_num
              << " more images to make the number of images is an integral multiple of batchsize[" << batch_size
              << "].";
    while (pad_num--) {
      image_paths.emplace_back(*image_paths.rbegin());
    }
  }
  return image_paths;
}

/**
  @return Returns resized image and scaling factors
 */
std::pair<cv::Mat, float> LetterBox(cv::Mat img, int dst_h, int dst_w, uint8_t pad_value) {
  float scaling_factors = std::min(1.0f * dst_h / img.rows, 1.0f * dst_w / img.cols);
  int unpad_h = std::floor(scaling_factors * img.rows);
  int unpad_w = std::floor(scaling_factors * img.cols);
  int pad_h = dst_h - unpad_h;
  int pad_w = dst_w - unpad_w;
  int pad_top = std::floor(pad_h / 2.0f);
  int pad_left = std::floor(pad_w / 2.0f);
  int pad_bottom = pad_h - pad_top;
  int pad_right = pad_w - pad_left;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(unpad_w, unpad_h));
  cv::Mat dst;
  cv::copyMakeBorder(resized, dst, pad_top, pad_bottom, pad_left, pad_right,
      cv::BORDER_CONSTANT, cv::Scalar(pad_value, pad_value, pad_value));
  return std::make_pair(dst, scaling_factors);
}


std::pair<cv::Mat, float> Preprocess(cv::Mat img, const magicmind::Dims &input_dim) {
    // NHWC order implementation. Make sure your model's input is in NHWC order.
  /*
     (x - mean) / std : This calculation process is performed at the first layer of the model,
     See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
  */
  // resize as latter box
  int h = input_dim[1];
  int w = input_dim[2];
  auto ret = LetterBox(img, h, w, 128);
  return ret;
}
