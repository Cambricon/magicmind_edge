#include <mm_runtime.h>
#include <cnrt.h>
#include <fstream>
#include <string>
#include <vector>
#include "../include/utils.hpp"

/**
 * @brief model's info
 * @param mm model
 */
void PrintModelInfo(magicmind::IModel *model)
{
  std::cout << "================== Model Info  ====================" <<std::endl;
  std::cout << "Input number : " << model->GetInputNum() << std::endl;
  for (int i = 0; i < model->GetInputNum(); ++i)
    std::cout << "input[" << i << "] : dimensions " << model->GetInputDimension(i)
              << ", data type [" << model->GetInputDataType(i) << "]" << std::endl;
  std::cout << "Output number : " << model->GetOutputNum() << std::endl;
  for (int i = 0; i < model->GetOutputNum(); ++i)
    std::cout << "output[" << i << "] : dimensions " << model->GetOutputDimension(i)
              << ", data type [" << model->GetOutputDataType(i) << "]" << std::endl;
}

std::string GetFileName(const std::string &abs_path) {
  auto slash_pos = abs_path.rfind('/');
  LOG_IF(FATAL, std::string::npos == slash_pos)
    << "[" << abs_path << "] is not an absolute path.";
  if (slash_pos == abs_path.size() - 1) {
    return "";
  }
  auto point_pos = abs_path.rfind('.');
  LOG_IF(FATAL, point_pos == slash_pos + 1)
    << "[" << abs_path << "] is not a file path.";
  return abs_path.substr(slash_pos + 1, point_pos - slash_pos - 1);
}

std::vector<std::string> LoadFileList(const std::string &path) {
  std::ifstream ifs(path);
  LOG_IF(FATAL, !ifs.is_open()) << "Open label file failed. path : "
      << path;
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(ifs, line)) labels.emplace_back(std::move(line));
  ifs.close();
  return labels;
}
