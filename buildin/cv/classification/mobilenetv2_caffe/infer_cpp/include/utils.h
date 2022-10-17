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

#define CHECK_STATUS(status)                           \
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
inline std::vector<int> sort_indexes(const std::vector<float> &v, bool reverse = false)
{
    std::vector<int> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
    if (reverse)
    {
        std::sort(idx.begin(), idx.end(),
                  [&v](int i1, int i2)
                  { return v[i1] > v[i2]; });
    }
    else
    {
        std::sort(idx.begin(), idx.end(),
                  [&v](int i1, int i2)
                  { return v[i1] < v[i2]; });
    }

    return idx;
}

inline std::string GetFileName(const std::string &abs_path) {
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

void PrintModelInfo(magicmind::IModel *model);

#endif // UTILS_HPP
