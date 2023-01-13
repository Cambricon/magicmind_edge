#include "pre_process.h"
#include "utils.h"
#include <glog/logging.h>

/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir, int batch_size, const std::string file_list) {
    char abs_path[PATH_MAX];
    if (realpath(image_dir.c_str(), abs_path) == NULL) {
        std::cout << "Get real image path in " << image_dir.c_str() << " failed...";
        exit(1);
    }
    std::string glob_path = std::string(abs_path);
    std::ifstream in(file_list);
    std::string image_name;
    std::vector<std::string> image_paths;
    std::string image_path;
    while(getline(in, image_name)) {
        image_path = glob_path + "/" +image_name;
        image_paths.push_back(image_path);
    }
    // pad to multiple of batch_size.
    // The program will stuck when the number of input images is not an integer multiple of the batch size
    size_t pad_num = batch_size - image_paths.size() % batch_size;
    if (pad_num != batch_size) {
        std::cout << "There are " << image_paths.size() << " images in total, add " << pad_num
            << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
        while (pad_num--)
            image_paths.emplace_back(*image_paths.rbegin());
    }
    return image_paths;
}

cv::Mat Preprocess(cv::Mat src_img)
{
    cv::resize(src_img, src_img, cv::Size(800, 288),cv::INTER_LINEAR);
    cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
    cv::Mat float_mat;
    src_img.convertTo(float_mat, CV_32FC3, 1.0/255);
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224,0.225);
    float_mat = float_mat -  mean;
    cv::divide(float_mat, std, float_mat);
    return float_mat;
}
