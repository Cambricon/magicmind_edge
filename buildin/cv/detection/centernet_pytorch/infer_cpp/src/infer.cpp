#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <sys/stat.h>
#include <memory>
#include <CLI11.hpp>
#include <cstring>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "../include/pre_process.hpp"
#include "../include/post_process.hpp"
#include "../include/utils.hpp"
using namespace magicmind;
using namespace std;
using namespace cv;

struct Args
{
  std::string magicmind_model = "../data/models/centernet_qint8_mixed_float16_1.mm";
  std::string image_dir = "../../../../../datasets/coco/";
  std::string file_list = "file_list.txt";
  std::string label_path = "../../../utils/coco.names";
  std::string output_path = "../data/images/";
  int image_num = 1;
  int batch_size = 1;
  int max_bbox_num = 100;
  float confidence_thresholds = 0.001;
  bool save_img = false;
  int device_id = 0;
};

int main(int argc, char **argv)
{
  Args args;
  CLI::App app{"centernet pytorch demo"};
  app.add_option("--magicmind_model", args.magicmind_model, "input mm model path")->check(CLI::ExistingFile);
  app.add_option("--image_dir", args.image_dir, "predict image file")->check(CLI::ExistingDirectory);
  app.add_option("--file_list", args.file_list, "file_list")->check(CLI::ExistingFile);
  app.add_option("--label_path", args.label_path, "coco.names")->check(CLI::ExistingFile);
  app.add_option("--output_path", args.output_path, "save img and txt file path. default: ../data/images/");
  app.add_option("--save_img", args.save_img, "save img or not. default: false");
  app.add_option("--confidence_thresholds", args.confidence_thresholds, "confidence thresholds. default: 0.001");
  app.add_option("--max_bbox_num", args.max_bbox_num, "confidence thresholds. default: 100");
  app.add_option("--image_num", args.image_num, "confidence thresholds. default: 1");
  app.add_option("--batch_size", args.batch_size, "batch_size. default: 1");
  app.add_option("--device_id", args.device_id, "device_id. default: 0");

  CLI11_PARSE(app, argc, argv);

  // 1. cnrt init
  LOG(INFO) << "Cnrt init..." << std::endl;
  MluDeviceGuard device_guard(args.device_id);
  cnrtQueue_t queue;
  CHECK_CNRT(cnrtQueueCreate, &queue);

  // 2.create model
  LOG(INFO) << "Load model..." << std::endl;
  auto model = CreateIModel();
  CHECK_PTR(model);
  MM_CHECK(model->DeserializeFromFile(args.magicmind_model.c_str()));
  PrintModelInfo(model);
  // 3. crete engine
  LOG(INFO) << "Create engine..." << std::endl;
  auto engine = model->CreateIEngine();
  CHECK_PTR(engine);

  // 4. create context
  LOG(INFO) << "Create context..." << std::endl;
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5. crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  // 6. input tensor memory alloc
  void *mlu_addr_ptr;
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  if (input_dim_vec[0] == -1)
  {
    input_dim_vec[0] = args.batch_size;
  }
  std::vector<magicmind::Dims> output_dims;
  for (size_t output_id = 0; output_id < model->GetOutputNum(); ++output_id)
  {
    auto output_dim_vec = model->GetOutputDimension(output_id).GetDims();
    if (output_dim_vec[0] == -1)
    {
      output_dim_vec[0] = args.batch_size;
    }
    output_dims.push_back(magicmind::Dims(output_dim_vec));
  }

  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);

  CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, input_tensors[0]->GetSize()));
  MM_CHECK(input_tensors[0]->SetData(mlu_addr_ptr));

  void *output_mlu_addr_ptr = nullptr;
  if (magicmind::Status::OK() == context->InferOutputShape(input_tensors, output_tensors)) 
  {
    for (size_t output_id = 0; output_id < model->GetOutputNum(); ++output_id) 
    {
      CNRT_CHECK(cnrtMalloc(&output_mlu_addr_ptr, output_tensors[output_id]->GetSize()));
      MM_CHECK(output_tensors[output_id]->SetData(output_mlu_addr_ptr));
    }
  }
  else 
  {
    output_tensors.clear();
    MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
  }

  // 7. load image
  LOG(INFO) << "================== Load Images ====================" << std::endl;
  std::vector<std::string> image_paths = LoadImages(args.image_dir, args.batch_size, args.image_num, args.file_list);
  if (image_paths.size() == 0)
  {
    LOG(INFO) << "No images found in dir [" << args.image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  LOG(INFO) << "Total images : " << image_num << std::endl;

  LOG(INFO) << "Start run..." << std::endl;
  std::vector<float *> net_outputs;
  for (size_t output_id = 0; output_id < output_dims.size(); output_id++)
  {
    float *data_ptr = new float[output_tensors[output_id]->GetSize() / sizeof(output_tensors[output_id]->GetDataType())];
    net_outputs.push_back(data_ptr);
  }
  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < image_num; i++)
  {
    string image_name = image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 12);
    LOG_EVERY_N(INFO, 100) << "Inference img : " << image_name << "\t\t\t" << i + 1 << "/" << image_num << std::endl;
    cv::Mat img = cv::imread(image_paths[i]);
    cv::Mat img_pro = Preprocess(img, input_dim);
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pro.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 8. compute
    MM_CHECK(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));
    // 9. copy out
    for (size_t output_id = 0; output_id < output_dims.size(); output_id++)
    {
      CNRT_CHECK(cnrtMemcpy(net_outputs[output_id], output_tensors[output_id]->GetMutableData(),
                            output_tensors[output_id]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    }

    // postprocess
    map<int, string> name_map = load_name(args.label_path);
    auto bboxes = Postprocess(net_outputs, output_dims, args.max_bbox_num, args.confidence_thresholds);
    // rescale bboxes to origin image.
    RescaleBBox(img, output_dims[0], bboxes, name_map, image_name, args.output_path);
    if (args.save_img)
    {
      // draw bboxes on original image and save it to disk.
      cv::Mat origin_img = img.clone();
      Draw(img, bboxes, name_map);
      cv::imwrite(args.output_path + "/" + image_name + ".jpg", img);
    }
  }
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
  LOG(INFO) << "E2E Execution time: " << execution_time.count() << "ms";
  LOG(INFO) << "E2E Throughput(1000 / execution time * image number): " << 1000 / execution_time.count() * image_num << "fps";
  // 10. destroy resource
  for (vector<float *>::const_iterator itr = net_outputs.begin(); itr != net_outputs.end(); ++itr)
  {
    delete *itr;
  }
  net_outputs.clear();
  for (auto tensor : input_tensors)
  {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : output_tensors)
  {
    if (output_mlu_addr_ptr != nullptr)
    {
        cnrtFree(tensor->GetMutableData());
    }
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();
  return 0;
}
