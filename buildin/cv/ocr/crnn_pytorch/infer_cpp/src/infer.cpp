#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <sys/stat.h>
#include <memory>
#include <CLI11.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <cstring>
#include <string>
#include "../include/pre_process.hpp"
#include "../include/post_process.hpp"
#include "../include/utils.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <chrono>

using namespace magicmind;
using namespace std;
using namespace cv;

struct Args
{
  std::string magicmind_model = "../data/models/crnn_qint8_mixed_float16_1.mm";
  std::string image_dir = "../../data/origin_img";
  std::string output_dir = "../data/images/";
  int batch_size = 1;
  int device_id = 0;
};

int main(int argc, char **argv)
{
  Args args;
  CLI::App app{"crnn pytorch demo"};
  app.add_option("--magicmind_model", args.magicmind_model, "input mm model path")->check(CLI::ExistingFile);
  app.add_option("--image_dir", args.image_dir, "predict image file")->check(CLI::ExistingFile);
  app.add_option("--output_dir", args.output_dir, "save img and txt file path. default: ../data/images/");
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
  PTR_CHECK(model);
  MM_CHECK_OK(model->DeserializeFromFile(args.magicmind_model.c_str()));
  PrintModelInfo(model);
  // 3. crete engine
  LOG(INFO) << "Create engine..." << std::endl;
  auto engine = model->CreateIEngine();
  PTR_CHECK(engine);

  // 4. create context
  LOG(INFO) << "Create context..." << std::endl;
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);
  auto context = engine->CreateIContext();
  PTR_CHECK(context);

  // 5. crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  // 6. input tensor memory alloc
  void *mlu_addr_ptr;
  MM_CHECK_OK(input_tensors[0]->SetDimensions(Dims({1,1,32,100})));
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
  MM_CHECK_OK(input_tensors[0]->SetData(mlu_addr_ptr));


  // 7. load image
  LOG(INFO) << "================== Load Images ====================" << std::endl;
  std::vector<std::string> image_paths = LoadImages(args.image_dir, args.batch_size);
  if (image_paths.size() == 0)
  {
    LOG(INFO) << "No images found in dir [" << args.image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  LOG(INFO) << "Total images : " << image_num << std::endl;

  LOG(INFO) << "Start run..." << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  
  for (int i = 0; i < image_num; i++)
  {
    std::string image_name = GetFileName(image_paths[i]);
    LOG_EVERY_N(INFO, 100) << "Inference img : " << image_name << "\t\t\t" << i + 1 << "/" << image_num << std::endl;
    cv::Mat img = cv::imread(image_paths[i], IMREAD_GRAYSCALE); // IMREAD_COLOR
    cv::Mat img_pro = Preprocess(img, input_dim, true);
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pro.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));
    // cout << "input_tensors[0]->GetSize()=" << input_tensors[0]->GetSize() << endl;
    // 8. compute
    output_tensors.clear();
    MM_CHECK_OK(context->Enqueue(input_tensors, &output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));
    // cout << "output_tensors[0]->GetSize()=" << output_tensors[0]->GetSize() << endl;

    vector<vector<float>> results;
    float *data_ptr = nullptr;
    data_ptr = (float *)malloc(output_tensors[0]->GetSize());
    CNRT_CHECK(cnrtMemcpy((void *)data_ptr, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));  
    
    // 9. copy out
    // postprocess
    string result = post_process(data_ptr);
    // rescale bboxes to origin image.
    // cout << "***result: " << result << endl;
    if (!args.output_dir.empty())
        {
            std::ofstream outfile;
            outfile.open(args.output_dir+"/"+"result_file.txt", ios::app);
            outfile << image_name << " " << result << endl;
            outfile.close();
        }
    free(data_ptr);  
  }
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
  LOG(INFO) << "E2E Execution time: " << execution_time.count() << "ms";
  LOG(INFO) << "E2E Throughput(1000 / execution time * image number): " << 1000 / execution_time.count() * image_num << "fps";
  // 10. destroy resource
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
