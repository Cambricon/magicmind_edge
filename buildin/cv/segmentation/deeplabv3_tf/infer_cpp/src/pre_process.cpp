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

cv::Mat Preprocess(cv::Mat src_img) {
    int src_h = src_img.rows;
    int src_w = src_img.cols;
    int dst_h = 513;
    int dst_w = 513;
    float ratio =  1.0 / 513 / std::max(float(src_h),float(src_w));
    cv::resize(src_img, src_img, cv::Size(dst_w, dst_h), cv::INTER_AREA);
    cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
    return src_img;
}
