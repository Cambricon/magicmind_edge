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

#include "coco_result_saver.h"
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
  std::string model_file = "../data/models/yolov3_qint8_mixed_float16_1.mm";
  std::string image_dir = "../../../../../datasets/coco/val2017";
  std::string name_file = "../../../utils/coco.names";
  std::string output_path = "../data/images/";
  std::string coco_result = "../data/result.json";
  bool save_img = false;
};

int main(int argc, char **argv)
{
  Args args;
  CLI::App app{"yolov3 pytorch demo"};
  app.add_option("--magicmind_model", args.model_file, "input mm model path")->check(CLI::ExistingFile);
  app.add_option("--image_dir", args.image_dir, "predict image file")->check(CLI::ExistingDirectory);
  app.add_option("--label_path", args.name_file, "name file")->check(CLI::ExistingFile);
  app.add_option("--output_dir", args.output_path, "output_dir");
  app.add_option("--coco_result", args.coco_result, "../data/result.json file");
  app.add_option("--save_img", args.save_img, "save img or not. default: false");
  CLI11_PARSE(app, argc, argv);
  // 1. cnrt init
  LOG(INFO) << "Cnrt init...";
  uint8_t device_id = 0 ;
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
  input_tensors[0]->SetDimensions(Dims({input_dim[0], input_dim[1], input_dim[2], input_dim[3]}));
  CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, input_tensors[0]->GetSize()));
  CHECK_STATUS(input_tensors[0]->SetData(mlu_addr_ptr));
  
  float *data_ptr = new float[output_tensors[0]->GetSize() / sizeof(output_tensors[0]->GetDataType())];
  if (data_ptr != NULL)
  {
    memset(data_ptr, 0, output_tensors[0]->GetSize() / sizeof(output_tensors[0]->GetDataType()));
  }
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

  auto labels = LoadLabels(args.name_file);
  std::map<int, std::string> name_map = LoadLabelName(args.name_file);
  LOG(INFO) << "Number of labels: " << labels.size() << std::endl;

  LOG(INFO) << "Start run..." << std::endl;
  auto start_time = std::chrono::steady_clock::now();

  COCOResultSaver coco_saver(args.coco_result, labels);
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
    int detection_num;
    CNRT_CHECK(cnrtMemcpy((void *)&detection_num, output_tensors[1]->GetMutableData(), output_tensors[1]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy((void *)data_ptr, output_tensors[0]->GetMutableData(), detection_num * 7 * 4, CNRT_MEM_TRANS_DIR_DEV2HOST));
    for (int j = 0; j < detection_num; ++j)
    {
      std::vector<float> result;
      float class_idx = *(data_ptr + 7 * j + 1); 
      float score = *(data_ptr + 7 * j + 2);
      float xmin = *(data_ptr + 7 * j + 3);
      float ymin = *(data_ptr + 7 * j + 4);
      float xmax = *(data_ptr + 7 * j + 5);
      float ymax = *(data_ptr + 7 * j + 6);
      result.push_back(class_idx);
      result.push_back(score);
      result.push_back(xmin);
      result.push_back(ymin);
      result.push_back(xmax);
      result.push_back(ymax);
      results.push_back(result);
    }

    std::vector<BBox> bboxes = PostProcess(img, input_dim[1], input_dim[2], results, name_map, image_name, args.output_path, args.save_img);
    if (args.coco_result.c_str()) 
    {
      coco_saver.Write(image_paths[i], bboxes);
    }
  }

  if (args.coco_result.c_str()) 
  {
    LOG(INFO) << args.coco_result.c_str() << " saved." ;
  }
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
  LOG(INFO) << "E2E Execution time: " << execution_time.count() << "ms";
  LOG(INFO) << "E2E Throughput(1000 / execution time * image number): " << 1000 / execution_time.count() * image_num << "fps";

  // 9. destroy resourc  
  if (data_ptr != NULL)
  {
    delete[] data_ptr;
    data_ptr = NULL;
  }
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
