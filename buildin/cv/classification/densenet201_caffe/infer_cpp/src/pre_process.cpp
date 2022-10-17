#include "pre_process.h"
#include "utils.h"
#include <glog/logging.h>


/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
std::vector<cv::String> LoadImages(const std::string image_dir,const int batch_size) {
  char abs_path[PATH_MAX];
  LOG(INFO)<< image_dir.c_str();
  LOG_IF(FATAL, !realpath(image_dir.c_str(), abs_path))
     << "Get real image path failed.";
  std::string glob_path = std::string(abs_path);
  std::vector<cv::String> jpg_paths, jpeg_paths, image_paths;
  cv::glob(glob_path + "/*.jpg", image_paths, false);
  cv::glob(glob_path + "/*.JPEG", image_paths, false);
  if (!jpg_paths.empty())
    image_paths.insert(jpg_paths.begin(), jpg_paths.end(), image_paths.end());
  if (!jpeg_paths.empty())
    image_paths.insert(jpeg_paths.begin(), jpeg_paths.end(), image_paths.end());
  // pad to multiple of batch_size.
  // The program will stuck when the number of input images is not an integer multiple of the batch size
  size_t pad_num = batch_size - image_paths.size() % batch_size;
  if (pad_num != batch_size) {
    LOG(INFO) << "There are " << image_paths.size() << " images in total, add " << pad_num
        << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
    while (pad_num--)
      image_paths.emplace_back(*image_paths.rbegin());
  }
  return image_paths;
}

cv::Mat Preprocess(cv::Mat img){
    size_t h = img.rows;
    size_t w = img.cols;
    float scale = h < w ? scale = 256. / h:256. / w;
    size_t new_h = h * scale;
    size_t new_w = w * scale;

    cv::resize(img, img, cv::Size(new_w, new_h));
    size_t left_x = int((new_w - 224) / 2);
    size_t top_y = int((new_h - 224) / 2);
    img = img(cv::Rect(left_x, top_y, 224, 224)).clone();
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    //img.convertTo(img, CV_32F);
    //cv::Scalar MEAN(103.939002991, 116.778999329, 123.680000305);
    //cv::Scalar STD(1.0/58.8235294117647, 1.0/58.8235294117647, 1.0/58.8235294117647);
    //img -= MEAN;
    //cv::multiply(img ,STD, img);
    return img;
}

