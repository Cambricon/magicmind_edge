#include <mm_runtime.h>
#include <cnrt.h>
#include <CLI11.hpp>
#include <cstring>
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pre_process.h"
#include "utils.h"

using namespace magicmind;
using namespace cv;

/**
 * @brief input params
 * model_file: Magicmind model path;
 * image_dir: input images path;
 * name_file: label of image;
 * output_path: the detection output path,include *.jpg;
 */
struct Args {
  std::string model_file = "../data/models/densenet201_qint8_mixed_float16_1.mm";
  std::string image_dir = "../../../../../datasets/imagenet/";
  std::string output_dir = "../data/images/";
};

int main(int argc, char **argv) {
  Args args;
  CLI::App app{"densenet201 caffe demo"};
  app.add_option("--magicmind_model", args.model_file, "input mm model path")
      ->check(CLI::ExistingFile);
  app.add_option("--image_dir", args.image_dir, "predict image file")
      ->check(CLI::ExistingDirectory);
  app.add_option("--output_dir", args.output_dir, "output path")->check(CLI::ExistingDirectory);
  CLI11_PARSE(app, argc, argv);

  // 1. cnrt init
  LOG(INFO) << "Cnrt init...";
  uint8_t device_id = 0;
  MluDeviceGuard device_guard(device_id);
  cnrtQueue_t queue;
  CHECK_CNRT(cnrtQueueCreate, &queue);

  // 2. create model
  LOG(INFO) << "Load model...";
  IModel *model = CreateIModel();
  model->DeserializeFromFile(args.model_file.c_str());
  PrintModelInfo(model);

  // 3.crete engine
  LOG(INFO) << "Create engine...";
  auto engine = model->CreateIEngine();
  CHECK_PTR(engine);

  // 4.create context
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5.crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);
  CHECK_STATUS(context->InferOutputShape(input_tensors, output_tensors));

  auto input_dim = model->GetInputDimension(0);

  // 6.input tensor memory alloc
  float *output_cpu_ptrs = (float *)malloc(output_tensors[0]->GetSize());

  for (auto tensor : input_tensors) {
    void *mlu_addr_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
    CHECK_STATUS(tensor->SetData(mlu_addr_ptr));
  }

  //   output tensor memory alloc
  for (auto tensor : output_tensors) {
    void *mlu_addr_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
    CHECK_STATUS(tensor->SetData(mlu_addr_ptr));
  }

  // 7. load image and label
  LOG(INFO) << "================== Load Images ====================";
  std::vector<cv::String> image_paths = LoadImages(args.image_dir, input_dim[0]);
  if (image_paths.size() == 0) {
    LOG(INFO) << "No images found in dir [" << args.image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  LOG(INFO) << "Total images : " << image_num << std::endl;
  LOG(INFO) << "Start run..." << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  for (int i = 0; i < image_num; i++) {
    if (!check_file_exist(image_paths[i])) {
      LOG(INFO) << "image file " + image_paths[i] + " not found.\n";
      exit(1);
    }
    Mat img = imread(image_paths[i]);
    if (img.empty()) {
      LOG(INFO) << "Failed to open image file " + image_paths[i];
      exit(1);
    }
    std::string image_name = GetFileName(image_paths[i]);
    LOG_EVERY_N(INFO, 100) << "Inference img: " << image_name << "\t\t\t" << i << "/" << image_num
                           << std::endl;
    img = Preprocess(img, input_dim);
    // 8. copy in
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img.data, input_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 9. compute
    CHECK_STATUS(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 10. copy out
    CNRT_CHECK(cnrtMemcpy(output_cpu_ptrs, output_tensors[0]->GetMutableData(),
                          output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    std::vector<float> output_data(
        (float *)output_cpu_ptrs,
        (float *)output_cpu_ptrs + (output_tensors[0]->GetSize() / sizeof(float)));
    std::vector<int> sorted_index = sort_indexes(output_data, true);
    if (!args.output_dir.empty()) {
      std::string save_path = args.output_dir + "/" + image_name + "_result.txt";
      std::ofstream ofs(save_path);
      LOG_IF(FATAL, !ofs.is_open()) << "Create file [" << save_path << "] failed.";
      for (int i = 0; i < 5; ++i) {
        ofs << sorted_index[i] << " ";
      }
      ofs.close();
    }
  }
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
  LOG(INFO) << "E2E Execution time: " << execution_time.count() << "ms";
  LOG(INFO) << "E2E Throughput(1000 / execution time * image number): "
            << 1000 / execution_time.count() * image_num << "fps";

  // 8. destroy resource
  free(output_cpu_ptrs);
  for (auto tensor : input_tensors) {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : output_tensors) {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();
  return 0;
}
