#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>
#include <mm_runtime.h>
#include <glog/logging.h>
#include <cnrt.h>
#include <vector>
#include "sys/stat.h"

#define CHECK_CNRT(FUNC, ...)                                                         \
  do                                                                                  \
  {                                                                                   \
    cnrtRet_t ret = FUNC(__VA_ARGS__);                                                \
    LOG_IF(FATAL, CNRT_RET_SUCCESS != ret)                                            \
        << "Call " << #FUNC << " failed. Ret code [" << static_cast<int>(ret) << "]"; \
  } while (0)

#define CHECK_PTR(ptr)                         \
  do                                           \
  {                                            \
    if (ptr == nullptr)                        \
    {                                          \
      LOG(INFO) << "mm failure " << std::endl; \
      abort();                                 \
    }                                          \
  } while (0)

#define CHECK_MM(FUNC, ...) do {                                \
    magicmind::Status ret = FUNC(__VA_ARGS__);                  \
    if ( !ret.ok())                                             \
    {                                                           \
	    std::cout << ret.error_message() << std::endl;      \
    }                                                           \
} while(0)

#define CHECK_STATUS(status)                               \
  do                                                   \
  {                                                    \
    auto ret = (status);                               \
    if (ret != magicmind::Status::OK())                \
    {                                                  \
      LOG(INFO) << "mm failure: " << ret << std::endl; \
      abort();                                         \
    }                                                  \
  } while (0)

class MluDeviceGuard {
 public:
  MluDeviceGuard(int device_id) {
    CHECK_CNRT(cnrtSetDevice, device_id);
  }
};  // class MluDeviceGuard

class Record
{
public:
  Record(std::string filename)
  {
    outfile.open(("output/" + filename).c_str(), std::ios::trunc | std::ios::out);
  }

  ~Record()
  {
    if (outfile.is_open())
      outfile.close();
  }

  void write(std::string line, bool print = false)
  {
    outfile << line << std::endl;
    if (print)
    {
      std::cout << line << std::endl;
    }
  }

private:
  std::ofstream outfile;
};

inline static void PrintModelInfo(magicmind::IModel *model) {
  LOG(INFO) << "==================Model info===================";
  LOG(INFO) << "Input number : " << model->GetInputNum();
  for (int i = 0; i < model->GetInputNum(); ++i)
    LOG(INFO) << "input[" << i << "] : dimensions "
              << model->GetInputDimension(i) << ", data type ["
              << model->GetInputDataType(i) << "]";
  LOG(INFO) << "Output number : " << model->GetOutputNum();
  for (int i = 0; i < model->GetOutputNum(); ++i)
    LOG(INFO) << "output[" << i << "] : dimensions "
              << model->GetOutputDimension(i) << ", data type ["
              << model->GetOutputDataType(i) << "]";
}

inline bool check_file_exist(std::string path)
{
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0)
  {
    if ((buffer.st_mode & S_IFDIR) == 0)
    {
      return true;
    }
    return false;
  }
  return false;
}

inline bool check_folder_exist(std::string path)
{
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0)
  {
    if ((buffer.st_mode & S_IFDIR) == 0)
    {
      return false;
    }
    return true;
  }
  return false;
}

// The implementation of this function shows the limitations of the current
// program on the model.
static bool CheckModel(magicmind::IModel *model) {
  if (model->GetInputNum() != 1) {
    LOG(ERROR) << "Input number is [" << model->GetInputNum() << "].";
    return false;
  }
  if (model->GetOutputNum() != 1) {
    LOG(ERROR) << "Output number is [" << model->GetOutputNum() << "].";
    return false;
  }
  if (model->GetInputDimension(0)[3] != 1) {
    LOG(ERROR) << "Input channels [" << model->GetInputDimension(0)[3] << "].";
    return false;
  }
  if (model->GetInputDataType(0) != magicmind::DataType::FLOAT32) {
    LOG(ERROR) << "Input data type is [" << model->GetInputDataType(0) << "].";
    return false;
  }
  if (model->GetOutputDataType(0) != magicmind::DataType::FLOAT32) {
    LOG(ERROR) << "Output data type is [" << model->GetOutputDataType(0)
               << "].";
    return false;
  }
  return true;
}
inline std::vector<cv::String> GetFileList(std::string rex_string,
                                           std::string dir) {
  char abs_path[PATH_MAX];
  CHECK_NE(true, dir.empty());
  LOG_IF(FATAL, !realpath(dir.c_str(), abs_path)) << "Get " + dir + " failed.";
  std::string glob_path = std::string(abs_path);
  std::vector<cv::String> file_paths;
  cv::glob(glob_path + rex_string, file_paths, false);
  return file_paths;
}

template <typename T>
std::vector<T> ReadFile(const std::string &path) {
  std::ifstream ifs(path);
  LOG_IF(FATAL, !ifs.is_open()) << "Open label file failed. path : " << path;
  std::vector<T> data;
  std::string line;
  while (std::getline(ifs, line)) {
    std::stringstream ss(line);
    T tmp = 0;
    ss >> tmp;
    data.emplace_back(tmp);
  }
  ifs.close();
  return data;
}

template <typename T>
std::vector<T> ReadBinFile(const std::string &path) {
  int size = 0;
  std::ifstream ifs(path, std::ifstream::binary);
  LOG_IF(FATAL, !ifs.is_open()) << "Open label file failed. path : " << path;
  ifs.seekg(0, ifs.end);
  size = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::vector<T> data(size / sizeof(T));
  ifs.read(reinterpret_cast<char *>(data.data()), size);
  ifs.close();
  return data;
}

template <typename T>
void WriteFile(const std::string &path, int count, T *ptr) {
  std::ofstream ofresult(path);
  for (int index = 0; index < count; index++) {
    ofresult << ptr[index] << "\n";
  }
  ofresult.close();
}

template <typename T>
void WriteBinFile(const std::string &path, int count, T *ptr) {
  std::ofstream ofresult(path);
  int size = count * sizeof(T);
  ofresult.write(reinterpret_cast<char *>(ptr), size);
  ofresult.close();
}



#endif // UTILS_HPP
