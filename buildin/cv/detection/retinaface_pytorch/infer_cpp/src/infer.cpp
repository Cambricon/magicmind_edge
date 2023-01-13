#include <sys/stat.h>
#include <memory>
#include <CLI11.hpp>
#include <string>
#include <cstring>
#include <chrono>
#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include "../include/pre_process.h"
#include "../include/post_process.h"
#include "../include/utils.h"
using namespace magicmind;
// init anchors for postprocessing. Call it once before postprocessing.
extern void InitAnchors(int height, int width);

/**
 * @brief input params
 * model_file: Magicmind model path;
 * image_dir: input images path;
 * output_path: the detection output path,include *.jpg;
 * batch_size: batch size;
 * conf: confidence_thresholds;
 * nms: nms_thresholds;
 * save_img: save detection img yes or not;
 * save_txt: save detection boxes in txt file;
 */
struct Args
{
  std::string model_file = "../data/models/retinaface_qint8_mixed_float16_1.mm";
  std::string image_dir = "../../../../../datasets/WIDER_val/images";
  std::string output_path = "../data/output/";
  int batch_size = 1;
  float conf = 0.02;
  float nms = 0.4;
  bool save_img = false;
  bool save_txt = false;
};

int main(int argc, char **argv)
{
  Args args;
  CLI::App app{"mobilenetssd pytorch demo"};
  app.add_option("--magicmind_model", args.model_file, "input mm model path")->check(CLI::ExistingFile);
  app.add_option("--image_dir", args.image_dir, "predict image file")->check(CLI::ExistingFile);
  app.add_option("--save_img", args.save_img, "save img or not. default: false");
  app.add_option("--save_txt", args.save_txt, "save detection out in txt file. default: true");
  app.add_option("--output_path", args.output_path, "save img and txt file path. default: ../data/images/");
  app.add_option("--batch_size", args.batch_size, "batch size. default: 1");
  app.add_option("--conf", args.conf, "confidence thresholds. default: 0.02");
  app.add_option("--nms", args.nms, "nms thresholds. default: 0.4");

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
  MM_CHECK(model->DeserializeFromFile(args.model_file.c_str()));
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
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  auto output_num = model->GetOutputNum();
  auto output_dims = model->GetOutputDimensions();
  if (input_dim_vec[0] == -1)
  {
    input_dim_vec[0] = args.batch_size;
  }
  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  InitAnchors(input_dim[1], input_dim[2]);
  input_tensors[0]->SetDimensions(input_dim);
  CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, input_tensors[0]->GetSize()));
  input_tensors[0]->SetData(mlu_addr_ptr);

  const int elem_data_count = input_tensors[0]->GetSize() / args.batch_size;
  const int loc_offset = output_dims[0].GetElementCount() / args.batch_size;
  const int conf_offset = output_dims[1].GetElementCount() / args.batch_size;
  const int landms_offset = output_dims[2].GetElementCount() / args.batch_size;
  uint8_t *input_data_ptr = new uint8_t[input_tensors[0]->GetSize()];

  // 7. load image
  LOG(INFO) << "================== Load Images ====================";
  std::vector<std::string> image_paths = LoadImages(args.image_dir, args.batch_size);
  if (image_paths.size() == 0)
  {
    LOG(INFO) << "No images found in dir [" << args.image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  LOG(INFO) << "Total images : " << image_num << std::endl;

  LOG(INFO) << "Start run..." << std::endl;
  std::vector<float *> net_outputs;
  std::vector<float> scaling_factors;
  std::vector<cv::Mat> ori_img;
  for (size_t output_id = 0; output_id < output_dims.size(); output_id++)
  {
    float *data_ptr = new float[output_tensors[output_id]->GetSize() / sizeof(output_tensors[output_id]->GetDataType())];
    net_outputs.push_back(data_ptr);
  }

  auto start_time = std::chrono::steady_clock::now();
  std::pair<cv::Mat, float> img_info;
  for (int i = 0; i < image_num;)
  {
    // 8. copy in
    memset(input_data_ptr, 0, sizeof(input_data_ptr));
    for (int bs = 0; bs < args.batch_size; bs++)
    {
      cv::Mat img = cv::imread(image_paths[i + bs]);
      img_info = Preprocess(img, input_dim);
      memcpy(input_data_ptr + bs * elem_data_count, img_info.first.data, elem_data_count * sizeof(uint8_t));
      ori_img.push_back(img);
      scaling_factors.push_back(img_info.second);
    }
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), input_data_ptr, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 9. compute
    output_tensors.clear();
    context->Enqueue(input_tensors, &output_tensors, queue);
    CNRT_CHECK(cnrtQueueSync(queue));

    // 10. copy out
    for (size_t output_id = 0; output_id < output_dims.size(); output_id++)
    {
      CNRT_CHECK(cnrtMemcpy(net_outputs[output_id], output_tensors[output_id]->GetMutableData(),
                            output_tensors[output_id]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    }

    // postprocess
    for (int bs = 0; bs < args.batch_size; bs++)
    {
      std::vector<float *> preds = {net_outputs[0] + bs * loc_offset,
                                    net_outputs[1] + bs * conf_offset, net_outputs[2] + bs * landms_offset};
      std::vector<BBox> bboxes = RetinafacePostprocess(ori_img[bs], scaling_factors[bs], input_dim,
                                                       preds, output_dims, args.conf, args.nms);

      auto slash_pos_end = image_paths[i + bs].rfind("/");
      std::string image_relative_path = image_paths[i + bs].substr(0, slash_pos_end);
      auto slash_pos_start = image_relative_path.rfind("/");
      std::string image_label_path = image_relative_path.substr(slash_pos_start, slash_pos_end);
      std::string image_name = GetFileName(image_paths[i + bs]);
      LOG_EVERY_N(INFO, 100) << "Inference img: " << image_name << "\t\t\t" << i + bs << "/" << image_num << std::endl;

      if (args.save_img)
      {
        // draw bboxes on original image and save it to disk.
        if (slash_pos_start != std::string::npos)
        {
          std::string mkdir_cmd = "mkdir -p " + args.output_path + "/images/" + image_label_path;
          if (-1 == system(mkdir_cmd.c_str()))
          {
            std::cout << "mkdir error" << std::endl;
            return 0;
          }
        }
        cv::Mat origin_img = ori_img[bs].clone();
        Draw(origin_img, bboxes);
        cv::imwrite(args.output_path + "/images/" + image_label_path + "/" + image_name + ".jpg", origin_img);
      }
      if (args.save_txt)
      {
        if (slash_pos_start != std::string::npos)
        {
          std::string mkdir_cmd = "mkdir -p " + args.output_path + "/pred_txts/" + image_label_path;
          if (-1 == system(mkdir_cmd.c_str()))
          {
            std::cout << "mkdir error" << std::endl;
            return 0;
          }
        }
        WritePreds(args.output_path + "/pred_txts/" + image_label_path, image_name, bboxes);
      }
    }
    ori_img.clear();
    net_outputs.clear();
    scaling_factors.clear();
    i += args.batch_size;
  }
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
  LOG(INFO) << "==================     End     ====================";
  LOG(INFO) << "E2E Execution time: " << execution_time.count() << "ms";
  LOG(INFO) << "E2E Throughput(1000 / execution time * image number): " << 1000 / execution_time.count() * image_num << "fps";

  // 10. destroy resource
  delete[] input_data_ptr;
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
