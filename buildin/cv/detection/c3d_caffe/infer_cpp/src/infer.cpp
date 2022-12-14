#include <gflags/gflags.h>
#include <glog/logging.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

#include "utils.hpp"
#include "pre_process.hpp"
#include "post_process.hpp"

using namespace std;
using namespace cv;

DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_int32(clip_step, -1, "The distance between each video clip, -1 means equal to sampling_rate * clip_length");
DEFINE_int32(sampling_rate, 2, "The sampling rate for video clips");
DEFINE_string(video_list, "", "Video list file path.");
DEFINE_int32(batch_size, 8, "batch_size");
DEFINE_string(dataset_dir, "", "dataset_dir");
DEFINE_string(output_dir,"", "output_dir");
DEFINE_string(name_file, "", "The label path");
DEFINE_string(result_file, "", "The classification results output file");
DEFINE_string(result_label_file, "", "The classification results label output file");
DEFINE_string(result_top1_file, "", "The classification results top1 output file");
DEFINE_string(result_top5_file, "", "The classification results top5 output file");


int main(int argc, char **argv) 
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Cnrt
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue)); //mm0.12 cnrtCreateQueue 

    // Model
    auto model = magicmind::CreateIModel();
    MM_CHECK(model->DeserializeFromFile(FLAGS_magicmind_model.c_str()));
    PrintModelInfo(model);

    // Engine
    magicmind::IModel::EngineConfig engine_config;
    engine_config.SetDeviceType("MLU");
    engine_config.SetConstDataInit(true);
    auto engine = model->CreateIEngine(engine_config);
    CHECK_VALID(engine);

    // Context
    auto context = engine->CreateIContext();
    CHECK_VALID(context);
    
    // Input And Output Malloc
    vector<magicmind::IRTTensor *> input_tensors, output_tensors;
    MM_CHECK(context->CreateInputTensors(&input_tensors));
    MM_CHECK(context->CreateOutputTensors(&output_tensors));

    // Set Input tensor size
    int img_w = 112;
    int img_h = 112;
    int clip_len = 8;
    int img_c = 3;
    int batch_size = FLAGS_batch_size;
    int clip_steps = (FLAGS_sampling_rate-1)*clip_len;
    int n_classes = 101;
    magicmind::Dims input_set_dims = magicmind::Dims({batch_size,clip_len,img_h,img_w,img_c});
    input_tensors[0]->SetDimensions(input_set_dims);

    //InferShape
    magicmind::Status status = context->InferOutputShape(input_tensors, output_tensors);
    LOG_IF(FATAL, !status.ok()) << "This program can not deal with the model which enabled [graph_shape_mutable]. mm err : " << status;

    // malloc
    void *mlu_input_addr_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_input_addr_ptr, input_tensors[0]->GetSize()));
    MM_CHECK(input_tensors[0]->SetData(mlu_input_addr_ptr));

    //Set Input data ptr
    float *input_data_ptr = new float[batch_size*clip_len*img_h*img_w*img_c];
    int total_data_nums = batch_size*clip_len*img_h*img_w*img_c;

    //Set Output data ptr
    float *output_data_ptr = new float[output_tensors[0]->GetSize()/sizeof(output_tensors[0]->GetDataType())];

    //Load Videos and Read Frames
    cout << "Loading Videos..." << endl;
    result* test;
    test = loadVideosAndLabels(FLAGS_video_list,FLAGS_name_file);

    if (test->video_paths.size() == 0) {
        cout << "No videos in video list[" << FLAGS_video_list << "].";
        return 0;
    }

    int total_videos = test->video_paths.size();
    int total_clips_id = 0;
    cv::VideoCapture capture;
    cout << "All Videos: " << total_videos << endl;
    cv::Mat frame;
    Record result_label(FLAGS_result_label_file);
    Record result_top1_file(FLAGS_result_top1_file);
    Record result_top5_file(FLAGS_result_top5_file);
    Record result_file(FLAGS_result_file);

    //start infer
    total_videos = (int)(total_videos/30);//???????????????CI-TEST ?????????????????????????????????
    cout << "Infering..." << endl;
    for ( int i = 0 ; i < total_videos; i++ )
    {
        std::string real_path = FLAGS_dataset_dir + '/' + test->video_paths[i];

        if ( ! capture.open(real_path) ){
            std::cout<<"Failed to open video: " << real_path << endl;
            break;
        }
        bool video_end =  false; //????????????????????????
        int clip_id = 0;//????????????????????????
        while ( ! video_end )
        {
            //pre_process
            int per_vaild_batch = 0; //??????batch_size????????????batch
            int pass_frames = 0;
            int _bs_index = 0;
            for ( _bs_index = 0; _bs_index < batch_size ; _bs_index++){
                int start = 0;
                while ( start < ( pass_frames + clip_len ) ) {
                    if ( capture.read(frame) ){
                        if (start >= pass_frames) {
                            cv::Mat ret = PreprocessImage(frame);
                            memcpy(input_data_ptr+(_bs_index*clip_len+(start-pass_frames))*img_c*img_h*img_w , \
                                    ret.data , img_c*img_h*img_w*magicmind::DataTypeSize(input_tensors[0]->GetDataType()));
                        }
                    }
                    else{
                        video_end = true;
                        break;
                    }
                    start ++;
                }
                if ( ! video_end ){
                    per_vaild_batch ++; //?????????bs*clip_len 1*8????????????+1 ??????batch 
                    pass_frames = clip_steps;
                    clip_id ++; 
                }
            }
            //compute
            CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(),input_data_ptr, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));
            output_tensors.clear();
            MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
            CNRT_CHECK(cnrtQueueSync(queue));
            CNRT_CHECK(cnrtMemcpy((void *)output_data_ptr, output_tensors[0]->GetMutableData(),output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
           //post process
            for (int _p = 0; _p < per_vaild_batch; _p++){
                int curr_clip_id = total_clips_id + clip_id - per_vaild_batch + _p ;
                std::vector<float> output_data((float *)output_data_ptr, (float *)output_data_ptr + (output_tensors[0]->GetSize() / sizeof(float)));
                std::vector<int> top5 = sort_indexes(output_data, true);

                auto pos = test->video_paths[i].find_last_of('/');
                string video_name = test->video_paths[i].substr(pos+1);
                std::string output_clip_name = FLAGS_output_dir + "/" + video_name + "_" + std::to_string(curr_clip_id) ;
                //?????????????????????
                if (!FLAGS_result_label_file.empty()) {
                    result_label.write("[" + std::to_string(curr_clip_id) + "]: " + std::to_string(test->name_to_id[getBaseName(test->video_paths[i])]), false);
                }
                //???????????????top1???
                if (!FLAGS_result_top1_file.empty()) {
                    result_top1_file.write("[" + std::to_string(curr_clip_id) + "]: " + std::to_string(top5[0]+1), false);
                }
                result_file.write("top5 result in " + video_name + ":", false);
                //???????????????top5??? key???id
                for (int j = 0 ; j < 5 ; j ++) {
                    if (!FLAGS_result_top5_file.empty()) {
                        result_top5_file.write("[" + std::to_string(curr_clip_id) + "]: " + std::to_string(top5[j]+1), false);
                    }
                    //???????????????top5??? key???image_name
                    if (!FLAGS_result_file.empty()) {
                        result_file.write(std::to_string(j) + " [" + test->id_to_name[top5[j]+1] + "]", false);
                    }
                }
            }
        }
        total_clips_id += clip_id;
    }
    cout << "Process Total:" << total_videos << " Videos and Total Clips:" << total_clips_id << endl;
    //end free resources
    CNRT_CHECK(cnrtQueueDestroy(queue));
    delete [] input_data_ptr;
    delete [] output_data_ptr;
    for (auto tensor : input_tensors) {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    for (auto tensor : output_tensors) {
        tensor->Destroy();
    }
    context->Destroy();
    engine->Destroy();
    model->Destroy();
    return 0;
}

