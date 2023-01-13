#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>
#include "sys/stat.h"
#include <mm_runtime.h>
#include <cnrt.h>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>


#define CHECK_CNRT(FUNC, ...) do {                                                                            \
    cnrtRet_t ret = FUNC(__VA_ARGS__);                                                                        \
    if ( ret != CNRT_RET_SUCCESS)                                                                             \
    {  std::cout << "Call " << #FUNC << " failed. Ret code [" << static_cast<int>(ret) << "]"  <<  std::endl; \
       abort();                                                                                               \
    }                                                                                                         \
} while(0)

class MluDeviceGuard {
 public:
  MluDeviceGuard(int device_id) {
    CHECK_CNRT(cnrtSetDevice, device_id);
  }
};  // class MluDeviceGuard

#define CHECK_MM(FUNC, ...) do {                                \
    magicmind::Status ret = FUNC(__VA_ARGS__);                  \
    if ( !ret.ok())                                             \
    {                                                           \
	    std::cout << ret.error_message() << std::endl;      \
    }                                                           \
} while(0)

#define MM_CHECK_OK(status)                                                                           \
    do                                                                                                \
    {                                                                                                 \
        auto ret = (status);                                                                          \
        if (ret != magicmind::Status::OK())                                                           \
        {                                                                                             \
            std::cout << "[" << __FILE__ << ":" << __LINE__ << "]  mm failure: " << ret << std::endl; \
            abort();                                                                                  \
        }                                                                                             \
    } while (0)

#define PTR_CHECK(ptr)                               \
    do                                               \
    {                                                \
        if (ptr == nullptr)                          \
        {                                            \
            std::cout << "mm failure " << std::endl; \
            abort();                                 \
        }                                            \
    } while (0)

class Record
{
public:
    Record(std::string filename){
        outfile.open(("output/" + filename).c_str(), std::ios::trunc | std::ios::out);
    }

    ~Record(){
        if(outfile.is_open())
            outfile.close();
    }

    void write(std::string line, bool print = false){
        outfile << line << std::endl;
        if (print)
        {
            std::cout << line << std::endl;
        }
    }

private:
    std::ofstream outfile;
};

inline bool check_file_exist(std::string path){
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0)
    {
        if ((buffer.st_mode & S_IFDIR) == 0)
            return true;
        return false;
    }
    return false;
}

inline bool check_folder_exist(std::string path){
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0)
    {
        if ((buffer.st_mode & S_IFDIR) == 0)
            return false;
        return true;
    }
    return false;
}
// Gets file paths from file list
std::vector<std::string> LoadFileList(const std::string &path);

void PrintModelInfo(magicmind::IModel *model);
// Gets file name without extension from the absolute path of the file
std::string GetFileName(const std::string &abs_path);
#endif