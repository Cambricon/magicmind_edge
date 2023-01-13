#include "../include/pre_process.hpp"
#include "../include/utils.hpp"

/**
 * @brief load all images(jpg) from image directory(args.image_dir)
 * @return Returns image paths
 */
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
// std::vector<cv::String> LoadImages(const std::string image_dir, const int batch_size)
// {
//   char abs_path[PATH_MAX];
//   if (realpath(image_dir.c_str(), abs_path) == NULL) {
//         std::cout << "Get real image path in " << image_dir.c_str() << " failed...";
//         exit(1);
//     }
//   std::string glob_path = std::string(abs_path);
//   std::vector<cv::String> image_paths;
//   cv::glob(glob_path + "/*.jpg", image_paths, false);
//   // pad to multiple of batch_size.
//   // The program will stuck when the number of input images is not an integer multiple of the batch size
//   size_t pad_num = batch_size - image_paths.size() % batch_size;
//   if (pad_num != batch_size)
//   {
//     LOG(INFO) << "There are " << image_paths.size() << " images in total, add " << pad_num
//         << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
//     while (pad_num--)
//     {
//       image_paths.emplace_back(*image_paths.rbegin());
//     }
//   }
//   return image_paths;
// }

cv::Mat Preprocess(cv::Mat img, const magicmind::Dims &input_dim, bool transpose){
    // NHWC order implementation. Make sure your model's input is in NHWC order.
    /*
       (x - mean) / std : This calculation process is performed at the first layer of the model,
       See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
    */
    //resize
    
    cv::resize(img, img, cv::Size(input_dim[2], input_dim[1]));
    // std::cout << "w:" << input_dim[2] << std::endl;
    // std::cout << "h:" << input_dim[1] << std::endl;
    img.convertTo(img, CV_32F);
    cv::Mat img_show;
    img.convertTo(img_show, CV_8U);            
    img /= 127.5;
    img -= 1.0;
    cv::Mat blob;
    if(transpose)
    {
        int c = img.channels();
        int h = img.rows;
        int w = img.cols;
        int sz[] = {1, c, h, w};
        blob.create(4, sz, img.depth()); // void create (int ndims, const int *sizes, int type)
        cv::Mat ch[1];
        for (int j = 0; j < c; j++)
        {
            ch[j] = cv::Mat(img.rows, img.cols, img.depth(), blob.ptr(0, j));
        }
        split(img, ch);
        img = blob;
    }
    return img_show;
}
