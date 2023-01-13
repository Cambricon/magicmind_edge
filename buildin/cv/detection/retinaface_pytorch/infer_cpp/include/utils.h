#ifndef UTILS_H_
#define UTILS_H_

#include <cnrt.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <mm_runtime.h>

#define CHECK_CNRT(FUNC, ...) do {                                                 \
  cnrtRet_t ret = FUNC(__VA_ARGS__);                                               \
  LOG_IF(FATAL, CNRT_RET_SUCCESS != ret)                                           \
    << "Call " << #FUNC << " failed. Ret code [" << static_cast<int>(ret) << "]";  \
} while(0)

#define CHECK_MM(FUNC, ...) do {              \
  magicmind::Status ret = FUNC(__VA_ARGS__);  \
  LOG_IF(FATAL, !ret.ok())                    \
    << ret.error_message() << std::endl;      \
} while(0)

#define CHECK_VALID(VALUE) do {           \
  auto t = (VALUE);                       \
  LOG_IF(FATAL, !t)                       \
    << #VALUE << " check valid failed.";  \
} while(0)

#define CHECK_PTR(ptr) do {                    \
    if (ptr == nullptr)                        \
    {                                          \
      LOG(INFO) << "mm failure " << std::endl; \
      abort();                                 \
    }                                          \
  } while (0)

#define MM_CHECK(status)                                     \
    do                                                       \
    {                                                        \
        auto ret = (status);                                 \
        if (ret != magicmind::Status::OK())                  \
        {                                                    \
            std::cout << "mm failure: " << ret << std::endl; \
            abort();                                         \
        }                                                    \
} while (0)

// Converts a void* to a pointer of the basic data type
template <typename DType> inline
DType* CastPtr(void *ptr) {
  return static_cast<DType *>(ptr);
}

template<typename T>
class BlockingQueue {
 public:
  void Stop() {
    running_ = false; 
    cond_.notify_all();
  }
  void Push(const T &data) {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      queue_.push(data);
    }
    cond_.notify_all();
  }
  bool PopFront(T *out) {
    std::unique_lock<std::mutex> lk(mtx_);
    cond_.wait(lk, [this] () {
      return !queue_.empty() || !running_;
    });
    if (!running_) return false;
    *out = std::move(queue_.front());
    queue_.pop();
    return true;
  }

 private:
  T invalid_;
  volatile bool running_ = true;
  std::queue<T> queue_;
  std::mutex mtx_;
  std::condition_variable cond_;
};  // class BlockingQueue

// In situations where condition_variable_any is used but no real locks are needed
class FakeLock {
 public:
  void lock() {}
  void unlock() {}
};  // class FakeLock

class MluDeviceGuard {
 public:
  MluDeviceGuard(int device_id) {
    CHECK_CNRT(cnrtSetDevice, device_id);
  }
};  // class MluDeviceGuard

// bounding box
struct BBox {
  float score;
  int cx, cy, w, h;
  std::vector<cv::Point2f> landms;
};  // struct BBox

// Gets file name without extension from the absolute path of the file
std::string GetFileName(const std::string &abs_path);

// Gets file paths from file list
std::vector<std::string> LoadFileList(const std::string &path);

void PrintModelInfo(magicmind::IModel *model);

#endif  // UTILS_H_
