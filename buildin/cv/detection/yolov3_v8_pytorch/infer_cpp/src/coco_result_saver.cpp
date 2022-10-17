#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"
#include "coco_result_saver.h"

static const std::unordered_map<std::string, int> kCOCOPaperCategoryIdMap = {
  {"person", 1},
  {"bicycle", 2},
  {"car", 3},
  {"motorcycle", 4},
  {"airplane", 5},
  {"bus", 6},
  {"train", 7},
  {"truck", 8},
  {"boat", 9},
  {"traffic light", 10},
  {"fire hydrant", 11},
  {"street sign", 12},
  {"stop sign", 13},
  {"parking meter", 14},
  {"bench", 15},
  {"bird", 16},
  {"cat", 17},
  {"dog", 18},
  {"horse", 19},
  {"sheep", 20},
  {"cow", 21},
  {"elephant", 22},
  {"bear", 23},
  {"zebra", 24},
  {"giraffe", 25},
  {"hat", 26},
  {"backpack", 27},
  {"umbrella", 28},
  {"shoe", 29},
  {"eye glasses", 30},
  {"handbag", 31},
  {"tie", 32},
  {"suitcase", 33},
  {"frisbee", 34},
  {"skis", 35},
  {"snowboard", 36},
  {"sports ball", 37},
  {"kite", 38},
  {"baseball bat", 39},
  {"baseball glove", 40},
  {"skateboard", 41},
  {"surfboard", 42},
  {"tennis racket", 43},
  {"bottle", 44},
  {"plate", 45},
  {"wine glass", 46},
  {"cup", 47},
  {"fork", 48},
  {"knife", 49},
  {"spoon", 50},
  {"bowl", 51},
  {"banana", 52},
  {"apple", 53},
  {"sandwich", 54},
  {"orange", 55},
  {"broccoli", 56},
  {"carrot", 57},
  {"hot dog", 58},
  {"pizza", 59},
  {"donut", 60},
  {"cake", 61},
  {"chair", 62},
  {"couch", 63},
  {"potted plant", 64},
  {"bed", 65},
  {"mirror", 66},
  {"dining table", 67},
  {"window", 68},
  {"desk", 69},
  {"toilet", 70},
  {"door", 71},
  {"tv", 72},
  {"laptop", 73},
  {"mouse", 74},
  {"remote", 75},
  {"keyboard", 76},
  {"cell phone", 77},
  {"microwave", 78},
  {"oven", 79},
  {"toaster", 80},
  {"sink", 81},
  {"refrigerator", 82},
  {"blender", 83},
  {"book", 84},
  {"clock", 85},
  {"vase", 86},
  {"scissors", 87},
  {"teddy bear", 88},
  {"hair drier", 89},
  {"toothbrush", 90},
  {"hair brus", 91}
};

// Returns -1 means unknown coco category id.
static inline
int LabelNameToCategoryId(const std::string &label_name) {
  auto iter = kCOCOPaperCategoryIdMap.find(label_name);
  return kCOCOPaperCategoryIdMap.end() == iter ? -1 : iter->second;
}

COCOResultSaver::COCOResultSaver(const std::string &output_file, const std::vector<std::string> &labels)
    : labels_(labels), writer_(stream_) {
  if (!output_file.empty()) {
    stream_.ofs.open(output_file);
    CHECK_EQ(true, stream_.ofs.is_open());
    writer_.StartArray();
  }
}

COCOResultSaver::~COCOResultSaver() {
  if (stream_.ofs.is_open()) {
    writer_.EndArray();
    writer_.Flush();
    stream_.ofs.close();
  }
}

void COCOResultSaver::Write(const std::string &image_path, const std::vector<BBox> &bboxes) {
  // image path to image id in coco style
  auto image_name = GetFileName(image_path);
  int image_id = -1;
  try {
    image_id = stoi(image_name);
  } catch (std::invalid_argument &e) {
    // maybe not a coco val image.
  }
  std::lock_guard<std::mutex> lk(mtx_);
  for (const auto &bbox : bboxes) Write(image_id, bbox);
}

void COCOResultSaver::Write(int image_id, const BBox &bbox) {
  std::string label_name = labels_[bbox.label];
  int category_id = LabelNameToCategoryId(label_name);
  if (-1 == category_id) {
    LOG(WARNING) << "Unknown label name [" << label_name << "].";
    return;
  }
  writer_.StartObject();
  writer_.Key("image_id");
  writer_.Int(image_id);
  writer_.Key("category_id");
  writer_.Int(category_id);
  writer_.Key("bbox");
  writer_.StartArray();
  writer_.Int(bbox.left);
  writer_.Int(bbox.top);
  writer_.Int(bbox.right - bbox.left);
  writer_.Int(bbox.bottom - bbox.top);
  writer_.EndArray();
  writer_.Key("score");
  // keep 5 decimal places, otherwise the writing may fail
  CHECK_EQ(true, writer_.Double(static_cast<int>(bbox.score * 100000) / 100000.0));
  writer_.EndObject();
}

