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

#include "utils.h"

using namespace magicmind;
using namespace cv;

/**
 * @brief input params
 * model_file: Magicmind model path;
 * input_data_dir: input data path;
 * softmax_output_dir: the detection output path, include *.jpg;
 */
struct Args
{
  std::string model_file = "../data/models/nnUNet_qint8_mixed_float16_1_0.mm";
  std::string input_data_dir = "../../../../../datasets/nnUNet_dataset/nn_UNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr1";
  std::string softmax_output_dir = "../data/softmax_output";
  std::string seg_output_dir = "../data/seg_output";
  int batch_size = 1;
};

void softmax_pixel(float *ptr, int h, int w, int c)
{
  for (int h_id = 0; h_id < h; h_id++)
  {
    for (int w_id = 0; w_id < w; w_id++)
    {
      float sum = 0.0;
      for (int c_id = 0; c_id < c; c_id++)
      {
        float value = ptr[h_id * w * c + w_id * c + c_id];
        ptr[h_id * w * c + w_id * c + c_id] = exp(value);
        sum += ptr[h_id * w * c + w_id * c + c_id];
      }
      for (int c_id = 0; c_id < c; c_id++)
      {
        ptr[h_id * w * c + w_id * c + c_id] /= sum;
      }
    }
  }
}

int main(int argc, char **argv)
{
  Args args;
  CLI::App app{"arcface pytorch demo"};
  app.add_option("--magicmind_model", args.model_file, "input mm model path")->check(CLI::ExistingFile);
  app.add_option("--input_data_dir", args.input_data_dir, "input data path")->check(CLI::ExistingDirectory);
  app.add_option("--softmax_output_dir", args.softmax_output_dir, "inference softmax_output path")->check(CLI::ExistingDirectory);
  app.add_option("--seg_output_dir", args.seg_output_dir, "inference seg *.jpg output path")->check(CLI::ExistingDirectory);
  app.add_option("--batch_size", args.batch_size, "batch_size");

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
  // Check if current program can deal with this model
  CHECK_EQ(CheckModel(model), true)
      << "Can not deal with this model.\n"
         "You should check your model with the "
         "following things:\n"
         "1. Make sure the data type of input is FLOAT.\n"
         "2. Make sure the input data is in NHWC order.\n"
         "3. Make sure the data type of output is FLOAT.\n"
         "4. Make sure the output data is in NHWC order.\n";

  // 3.crete engine
  LOG(INFO) << "Create engine...";
  auto engine = model->CreateIEngine();
  CHECK_PTR(engine);
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);

  // 4.create context
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5.crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  auto output_dim_vec = model->GetOutputDimension(0).GetDims();
  auto batch_size = args.batch_size;
  if (input_dim_vec[0] != -1 ) {
   CHECK_STATUS(context->InferOutputShape(input_tensors, output_tensors));
  }
  if (input_dim_vec[0] == -1) {
    input_dim_vec[0] = batch_size;
  }
  if (output_dim_vec[0] == -1) {
    output_dim_vec[0] = batch_size;
  }
  auto input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);
  auto output_dim = magicmind::Dims(output_dim_vec);
  output_tensors[0]->SetDimensions(output_dim);

  // 6.input tensor memory alloc
  for (auto tensor : input_tensors)
  {
    void *mlu_addr_ptr;
    LOG(INFO) << "input size:"<<tensor->GetSize();
    CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
    CHECK_STATUS(tensor->SetData(mlu_addr_ptr));
  }

  //   output tensor memory alloc in device
  for (auto tensor : output_tensors)
  {
    LOG(INFO) << "output size:"<< tensor->GetSize();
    void *mlu_addr_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
    CHECK_STATUS(tensor->SetData(mlu_addr_ptr));
  }
  // output tensor malloc in host
  float *output_cpu_ptrs = (float *)malloc(output_tensors[0]->GetSize());

  // 7. load image
  LOG(INFO) << "================== Load Images ====================";
  std::vector<cv::String> file_lists = GetFileList("/*data", args.input_data_dir);
  LOG(INFO) << "File list size: " << file_lists.size();
  LOG(INFO) << "Start run..." << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  uint64_t img_num = 0;
  std::vector<std::vector<int>> data_shapes;
  for (int file_index = 0; file_index < file_lists.size(); ++file_index)
  {
    auto data_shape_path = file_lists[file_index] + "_shape_info";
    std::vector<int> data_shape = ReadBinFile<int>(data_shape_path);
    img_num += (data_shape[1] * 4);
    data_shapes.emplace_back(std::move(data_shape));
  }

  for (int i = 0; i < file_lists.size(); i++)
  {
    std::vector<int> data_shape = data_shapes[i];
    std::string data_path = file_lists[i];
    std::vector<float> data = ReadBinFile<float>(data_path);
    float *data_ptr = data.data();
    int image_elemt_size = (int)(data_shape[2] * data_shape[3]);
    for (int index = 0; index < data_shape[1]; index++)
    {
      auto img_data = data_ptr + index * image_elemt_size;
      cv::Mat img(data_shape[2], data_shape[3], CV_32FC1, img_data);
      cv::Mat padded_img;
      int h = std::max((int64_t)data_shape[2], input_dim[1]);
      int w = std::max((int64_t)data_shape[3], input_dim[2]);
      int pad_h_before = (int)((h - data_shape[2]) / 2);
      int pad_h_after = (int)(h - data_shape[2] - pad_h_before);
      int pad_w_before = (int)((w - data_shape[3]) / 2);
      int pad_w_after = (int)(w - data_shape[3] - pad_w_before);
      cv::copyMakeBorder(img, padded_img, pad_h_before, pad_h_after,
                         pad_w_before, pad_w_after, cv::BORDER_CONSTANT,
                         cv::Scalar(0));

      for (int mirror_index = 0; mirror_index < batch_size; mirror_index++)
      {
        cv::Mat rgb(input_dim[1], input_dim[2], CV_32FC1);
        if (mirror_index % batch_size== 1)
        {
          cv::flip(padded_img, rgb, 0);
        }
        else if (mirror_index % batch_size == 2)
        {
          cv::flip(padded_img, rgb, 1);
        }
        else if (mirror_index % batch_size == 3)
        {
          cv::flip(padded_img, rgb, -1);
        }
        else
        {
          padded_img.copyTo(rgb);
        }
        // 8. copy in
        CNRT_CHECK(cnrtMemcpy(((float *)input_tensors[0]->GetMutableData()) + mirror_index * input_tensors[0]->GetSize() / batch_size / sizeof(float),
                              rgb.data, input_tensors[0]->GetSize() / batch_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
      }
      //  9. compute
      CHECK_STATUS(context->Enqueue(input_tensors, output_tensors, queue));
      CNRT_CHECK(cnrtQueueSync(queue));
      std::vector<float *> ptrs;
      std::vector<cv::Mat> imgs;
      int elem_data_count = output_dim.GetElementCount() / batch_size;
      for (int batch_id = 0; batch_id < batch_size; batch_id++)
      {
        ptrs.emplace_back(((float *)output_cpu_ptrs) + batch_id * elem_data_count);
        cv::Mat img(output_dim[1], output_dim[2], CV_32FC2, ptrs[batch_id]);
        imgs.emplace_back(std::move(img));
      }
      // 10. copy out
      CNRT_CHECK(cnrtMemcpy(output_cpu_ptrs, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
      cv::Mat tmp_img;
      for (int batch_id = 0; batch_id < batch_size; batch_id++)
      {
        softmax_pixel(ptrs[batch_id], output_dim[1], output_dim[2],
                      output_dim[3]);
        if (batch_id == 1)
        {
          cv::flip(imgs[batch_id], tmp_img, 0);
          cv::addWeighted(imgs[0], 0.25, tmp_img, 0.25, 0.0, imgs[0]);
        }
        else if (batch_id == 2)
        {
          cv::flip(imgs[batch_id], tmp_img, 1);
          cv::addWeighted(imgs[0], 1, tmp_img, 0.25, 0.0, imgs[0]);
        }
        else if (batch_id == 3)
        {
          cv::flip(imgs[batch_id], tmp_img, -1);
          cv::addWeighted(imgs[0], 1, tmp_img, 0.25, 0.0, imgs[0]);
        }
      }
      cv::Mat crop_img = imgs[0](cv::Rect(pad_w_before, pad_h_before, data_shape[3], data_shape[2])).clone();
      float *crop_data = (float *)crop_img.data;
      int count = data_shape[2] * data_shape[3] * 2;
      std::string output_path = data_path + "_output_" + std::to_string(index);
      int pos = data_path.find_last_of('/');
      std::string data_file_name(data_path.substr(pos + 1));
      if (!args.seg_output_dir.empty())
      {
        // compute argmax
        std::vector<uint8_t> argmax_result;
        for (int k = 0; k < count / 2; k++)
        {
          if (crop_data[2 * k] > crop_data[2 * k + 1])
          {
            argmax_result.push_back((uint8_t)0);
          }
          else
          {
            argmax_result.push_back((uint8_t)255);
          }
        }
        cv::Mat seg_img(data_shape[2], data_shape[3], CV_8UC1, argmax_result.data());
        std::string save_path = args.seg_output_dir + "/" +
                                data_file_name + "_output_" +
                                std::to_string(index) + "_seg.jpg";
        cv::imwrite(save_path, seg_img);
      }
      if (!args.softmax_output_dir.empty())
      {
        std::string save_path = args.softmax_output_dir + "/" +
                                data_file_name + "_output_" +
                                std::to_string(index);
        WriteBinFile(save_path, count, crop_data);
      }
    }
  }
  LOG(INFO) << "All images processed...";
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time =
      end_time - start_time;
  LOG(INFO) << "Execution time: " << execution_time.count() << "ms";
  LOG(INFO) << "Throughput(1000 / execution time * image number): "
            << 1000 / execution_time.count() * img_num << "fps";

  // 8. destroy resource
  free(output_cpu_ptrs);
  for (auto tensor : input_tensors)
  {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : output_tensors)
  {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();
  return 0;
}
