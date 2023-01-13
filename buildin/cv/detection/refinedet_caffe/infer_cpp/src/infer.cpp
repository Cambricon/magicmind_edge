#include <mm_runtime.h>
#include <cnrt.h>
#include <sys/stat.h>
#include <memory>
#include <CLI11.hpp>
#include <string>
#include <cstring>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "pre_process.h"
#include "post_process.h"
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
struct Args
{
  std::string model_file = "../data/models/refinedet_qint8_mixed_float16_1.mm";
  std::string image_dir = "../../../../../datasets/VOC2012/JPEGImages/";
  std::string output_path = "../data/images/";
  float conf = 0.001;
  bool save_img = false;
  bool save_txt = true;
};

// save detection results to disk (txt)
void WriteDetectionResults(const std::string image_name, const std::vector<BBox> &bboxes, const std::string output_path)
{
  std::string output_file_path = output_path + "/" + image_name + ".txt";
  std::ofstream ofs(output_file_path);
  CHECK_EQ(ofs.is_open(), true) << "Open/Create file [" << output_file_path << "] failed.";
  for (auto bbox : bboxes)
  {
    ofs << bbox.label << " " << bbox.score << " " << bbox.left << " " << bbox.top << " "
        << bbox.right << " " << bbox.bottom << std::endl;
  }
  ofs.close();
}

int main(int argc, char **argv)
{
  Args args;
  CLI::App app{"refinedet pytorch demo"};
  app.add_option("--magicmind_model", args.model_file, "input mm model path")->check(CLI::ExistingFile);
  app.add_option("--image_dir", args.image_dir, "predict image file")->check(CLI::ExistingDirectory);
  app.add_option("--save_img", args.save_img, "save img or not. default: false");
  app.add_option("--save_txt", args.save_txt, "save detection out in txt file. default: true");
  app.add_option("--output_path", args.output_path, "save img and txt file path. default: ../data/images/");
  app.add_option("--conf", args.conf, "confidence thresholds. default: 0.001");
  CLI11_PARSE(app, argc, argv);
  // 1. cnrt init
  LOG(INFO) << "Cnrt init...";
  uint8_t device_id = 0;
  MluDeviceGuard device_guard(device_id);
  cnrtQueue_t queue;
  CHECK_CNRT(cnrtQueueCreate, &queue);
  
  // 2. create model
  LOG(INFO) << "Load model...";
  auto model = CreateIModel();
  CHECK_PTR(model);
  CHECK_STATUS(model->DeserializeFromFile(args.model_file.c_str()));
  PrintModelInfo(model);

  // 3. crete engine
  LOG(INFO) << "Create engine...";
  auto engine = model->CreateIEngine();
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);
  CHECK_PTR(engine);

  // 4. create context
  LOG(INFO) << "Create context...";
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5. crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  // 6. input tensor memory alloc
  void *mlu_addr_ptr;
  auto input_dim = model->GetInputDimension(0);
  auto output_num = model->GetOutputNum();
  auto output_dim = model->GetOutputDimension(0);
  int batch_size = input_dim[0];

  input_tensors[0]->SetDimensions(Dims({input_dim[0], input_dim[1], input_dim[2], input_dim[3]}));
  CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, input_tensors[0]->GetSize()));
  CHECK_STATUS(input_tensors[0]->SetData(mlu_addr_ptr));

  // 7. load image
  LOG(INFO) << "================== Load Images ====================";
  std::vector<cv::String> image_paths = LoadImages(args.image_dir, input_dim[0]);
  if (image_paths.size() == 0)
  {
    LOG(INFO) << "No images found in dir [" << args.image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  LOG(INFO) << "Total images : " << image_num << std::endl;

  LOG(INFO) << "Start run..." << std::endl;
  auto start_time = std::chrono::steady_clock::now();

  static constexpr int bbox_size = 7;
  for (int i = 0; i < image_num; i++)
  {
    std::string image_name = GetFileName(image_paths[i]);

    LOG_EVERY_N(INFO, 100) << "Inference img: " << image_name << "\t\t\t" << i << "/" << image_num << std::endl;

    Mat img = imread(image_paths[i]);
    Mat img_pre = Preprocess(img, input_dim[1], input_dim[2]);
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pre.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 8. compute
    output_tensors.clear();
    CHECK_STATUS(context->Enqueue(input_tensors, &output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 9. copy out
    std::vector<std::vector<float>> results;

    for (uint32_t j = 0; j < output_num; j++)
    {
      int detection_num = output_tensors[j]->GetDimensions()[0];
      // get output data from tensor

      auto preds_dim = output_tensors[j]->GetDimensions();
      std::shared_ptr<void> preds_ptr = nullptr;
      std::vector<std::vector<BBox>> bboxes;
      float *output_cpu_ptrs = new float[output_tensors[j]->GetSize() / sizeof(output_tensors[j]->GetDataType())];
      if (output_cpu_ptrs != NULL)
      {
        memset(output_cpu_ptrs, 0, output_tensors[j]->GetSize() / sizeof(output_tensors[j]->GetDataType()));
      }
      if (output_tensors[j]->GetMemoryLocation() == magicmind::TensorLocation::kHost)
      {
        // memory in host
        bboxes = PostProcess(img, batch_size, preds_dim, args.conf, (float *)output_tensors[j]->GetMutableData(), image_name, args.output_path, args.save_img);
      }
      else if (output_tensors[j]->GetMemoryLocation() == magicmind::TensorLocation::kMLU)
      {
        // memory in device
        int64_t outsize = model->GetOutputDimension(j).GetElementCount() * magicmind::DataTypeSize(model->GetOutputDataType(j));
        CHECK_CNRT(cnrtMemcpy, output_cpu_ptrs, output_tensors[j]->GetMutableData(),
                   output_tensors[j]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST);
        bboxes = PostProcess(img, batch_size, preds_dim, args.conf, (float *)output_cpu_ptrs, image_name, args.output_path, args.save_img);
      }
      else
      {
        LOG(FATAL) << "Invalid memory location.";
      }
      if (args.save_txt)
      {
        WriteDetectionResults(image_name, bboxes[j], args.output_path);
      }
      if (output_cpu_ptrs != NULL)
      {
        delete[] output_cpu_ptrs;
        output_cpu_ptrs = NULL;
      }
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
  LOG(INFO) << "E2E Execution time: " << execution_time.count() << "ms";
  LOG(INFO) << "E2E Throughput(1000 / execution time * image number): " << 1000 / execution_time.count() * image_num << "fps";

  // 9. destroy resource
  for (auto tensor : input_tensors)
  {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : output_tensors)
  {
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();
  return 0;
}
