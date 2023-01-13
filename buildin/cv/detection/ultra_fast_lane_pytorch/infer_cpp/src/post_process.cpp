#include <glog/logging.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"
#include "post_process.h"
#include "utils.h"

static const int h_samples[] = {160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
                                270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
                                430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
                                590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710};

template <class T>
int lengths(T &arr)
{
    return sizeof(arr) / sizeof(arr[0]);
}

void PostProcess(int32_t *output_data_ptr, const std::string output_dir, const std::string name, const magicmind::Dims &output_dim)
{
    std::string json_path = output_dir + "/tusimple_eval_tmp.0.txt";
    std::ofstream fout(json_path, std::ios::out | std::ios::app);
    fout << "{\"lanes\":[";
    int output_h = output_dim[0];
    int output_w = output_dim[1];
    for (int i = 0; i < output_h; i++)
    {
        rapidjson::Value temp(rapidjson::kArrayType);
        fout << "[";
        for (int j = 0; j < output_w; j++)
        {
            fout << ((int32_t *)output_data_ptr)[i * output_w + j];
            if (j != output_w - 1)
                fout << ", ";
        }
        if (i != output_h - 1)
        {
            fout << "], ";
        }
        else
        {
            fout << "]";
        }
    }
    fout << "], \"h_samples\": [";
    int length = lengths(h_samples);
    for (int i = 0; i < length; i++)
    {
        fout << h_samples[i];
        if (i != length - 1)
            fout << ", ";
    }
    fout << "], \"raw_file\": \"" << name << "\", \"run_time\": 10}" << std::endl;
    fout.close();
}
