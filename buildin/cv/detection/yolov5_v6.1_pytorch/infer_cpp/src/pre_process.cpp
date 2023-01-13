#include "pre_process.h"
#include "utils.h"
#include <glog/logging.h>

std::vector<cv::Mat> LoadCocoImages(std::string coco_path, int count)
{
    std::vector<cv::Mat> imgs;
    std::string image_path = coco_path + "/images";
    if (!check_folder_exist(image_path))
    {
        LOG(INFO) << "image folder: " + image_path + " does not exist.\n";
        exit(0);
    }
    std::vector<cv::String> image_files;
    cv::glob(image_path + "/*.jpg", image_files);
    int current_count = 0;
    for (int i = 0; i < count; ++i)
    {
        cv::Mat img = cv::imread(image_files[i]);
        if (img.empty())
        {
            LOG(INFO) << "failed to load image " + image_files[i] + ".\n";
            exit(0);
        }
        imgs.push_back(img);
    }
    if (imgs.empty())
    {
        LOG(INFO) << " image folder: " + image_path + "not contian jpg file.\n";
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

cv::Mat Preprocess(cv::Mat src_img, bool transpose, bool normlize, bool swapBR, int depth)
{
    int src_h = src_img.rows;
    int src_w = src_img.cols;
    int dst_h = 640;
    int dst_w = 640;
    float ratio = std::min(float(dst_h) / float(src_h), float(dst_w) / float(src_w));
    int unpad_h = std::floor(src_h * ratio);
    int unpad_w = std::floor(src_w * ratio);
    if (ratio != 1)
    {
        int interpolation;
        if (ratio < 1)
        {
            interpolation = cv::INTER_AREA;
        }
        else
        {
            interpolation = cv::INTER_LINEAR;
        }
        cv::resize(src_img, src_img, cv::Size(unpad_w, unpad_h), interpolation);
    }

    int pad_t = std::floor((dst_h - unpad_h) / 2);
    int pad_b = dst_h - unpad_h - pad_t;
    int pad_l = std::floor((dst_w - unpad_w) / 2);
    int pad_r = dst_w - unpad_w - pad_l;

    cv::copyMakeBorder(src_img, src_img, pad_t, pad_b, pad_l, pad_r, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    if (normlize)
    {
        src_img.convertTo(src_img, CV_32F);
        cv::Scalar std(0.00392, 0.00392, 0.00392);
        cv::multiply(src_img, std, src_img);
    }
    if (swapBR)
    {
        cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
    }

    if (src_img.depth() != depth)
    {
        src_img.convertTo(src_img, depth);
    }

    cv::Mat blob;
    if (transpose)
    {
        int c = src_img.channels();
        int h = src_img.rows;
        int w = src_img.cols;
        int sz[] = {1, c, h, w};
        blob.create(4, sz, depth);
        cv::Mat ch[3];
        for (int j = 0; j < c; j++)
        {
            ch[j] = cv::Mat(src_img.rows, src_img.cols, depth, blob.ptr(0, j));
        }
        cv::split(src_img, ch);
    }
    else
    {
        blob = src_img;
    }
    return blob;
}
