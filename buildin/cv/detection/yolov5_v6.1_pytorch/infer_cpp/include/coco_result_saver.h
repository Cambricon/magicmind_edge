#ifndef COCO_RESULT_SAVER_H_
#define COCO_RESULT_SAVER_H_

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include "utils.h"

struct StreamImpl {
  using Ch = char;
  std::ofstream ofs;
  void Put(Ch ch) { ofs.put(ch); }
  void Flush() { ofs.flush(); }
};

/**
 * Save detections in COCO style JSON format.
 * This is a thread-safe implemention.
 */
class COCOResultSaver {
 public:
  /**
   * @param output_path[in]: output file path
   * @param labels[in]: label names
   */
  explicit COCOResultSaver(const std::string &output_path, const std::vector<std::string> &labels);
  ~COCOResultSaver();
  /**
   * @brief Write detection results for image
   * @param image_path[in]: image path
   * @param bboxes[in]: detection result
   */
  void Write(const std::string &image_path, const std::vector<BBox> &bboxes);

 private:
  // image_id : COCO style image id, gets from image_path
  void Write(int image_id, const BBox &bbox);

 private:
  std::mutex mtx_;
  std::vector<std::string> labels_;
  StreamImpl stream_;
  rapidjson::Writer<StreamImpl> writer_;
};  // class COCOResultSaver

#endif  // COCO_RESULT_SAVER_H_

