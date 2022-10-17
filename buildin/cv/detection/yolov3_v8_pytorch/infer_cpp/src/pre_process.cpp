#include "pre_process.h"
#include "utils.h"
#include <glog/logging.h>

std::vector<cv::Mat> LoadCocoImages(std::string coco_path, int count) {
    std::vector<cv::Mat> imgs;
    std::string image_path = coco_path + "/images";
    if (!check_folder_exist(image_path)){
        LOG(INFO) << "image folder: " + image_path + " does not exist.\n";
        exit(0);
    }
    std::vector<cv::String> image_files;
    cv::glob(image_path + "/*.jpg", image_files);
    int current_count = 0;
    for (int i = 0; i < count; ++i){
        cv::Mat img = cv::imread(image_files[i]);
        if(img.empty()){
            LOG(INFO) << "failed to load image " + image_files[i] + ".\n";
            exit(0);
        }
        imgs.push_back(img);
    }
    if(imgs.empty()){
        LOG(INFO) <<" image folder: " + image_path + "not contian jpg file.\n";
        exit(0);
    }
    return imgs;
}

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

cv::Mat Preprocess(cv::Mat img, int dst_h, int dst_w, bool transpose, bool normlize) {

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
    cv::copyMakeBorder(resized, dst, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::Mat rgb(dst_h, dst_w, CV_8UC3);
    cv::cvtColor(dst, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}
