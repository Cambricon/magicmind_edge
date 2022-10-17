
#include <iostream>
#include <string>
#include <algorithm>
#include "pre_process.hpp"
#include "utils.hpp"

static int img_w = 112;
static int img_h = 112;
static int resize_width = 171;
static int resize_height = 128;

void stripWhiteSpace(string &str){
	if (str == ""){
		return;
	}
 
	string::iterator cur_it;
	cur_it = str.begin();
 
	while (cur_it != str.end()){
		if (((*cur_it) != '\t') && ((*cur_it) != ' ')){
			break;
		}
		else{
			cur_it = str.erase (cur_it);
		}
	}
 
	cur_it = str.begin();
 
	while (cur_it != str.end())
	{
		if ((*cur_it) == '\n' || (*cur_it) == '\r'){
			cur_it = str.erase (cur_it);
		}
		else{
			cur_it++;
		}
	}
}


result *  loadVideosAndLabels(const std::string &path,const std::string &label_file) {
    result *test=new result;
    //Load Videos
    std::ifstream ifs(path);
    std::vector<std::string> videos;
    std::string line;
    if ( ! ifs.is_open() ){
        std::cout << "Open file failed: " << path << std::endl;
    }
    else{
        while (std::getline(ifs, line)) videos.emplace_back(line);
        ifs.close();
    }

    //Get category-id map
    int id = 0;
    std::string name;
    std::ifstream in_video(label_file);
    if (!in_video){
        std::cout<<"failed to load label_file: " + label_file + ".\n";
        exit(0);
    }
    while (getline(in_video, line))
    {
        int loc = line.find(" ");
        id = std::stoi(line.substr(0, loc));
        name = line.substr(loc+1);
        stripWhiteSpace(name);
        test->name_to_id[name] = id;
        test->id_to_name[id]=name;
    }

    for (int i = 0; i < videos.size(); i++){
        //ApplyEyeMakeup/v_ApplyEyeMakeup_g03_c02.avi
        stripWhiteSpace(videos[i]);
        int first_ = videos[i].find('_');
        std::string sub_1_video = videos[i].substr(first_+1);
        int second_ = sub_1_video.find('_');
        name = sub_1_video.substr(0,second_);
        test->video_paths.push_back(videos[i]);
        test->labels.push_back(test->name_to_id[name]);
    }
    return test;
}

std::string getBaseName(const string& fullname){
    int first_ = fullname.find('_');
    std::string sub_1_video = fullname.substr(first_+1);
    int second_ = sub_1_video.find('_');
    std::string name = sub_1_video.substr(0,second_);
    return name;
}

cv::Mat PreprocessImage(cv::Mat img) {
  int h = img_h;
  int w = img_w;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_width, resize_height));
  // center crop
  int left = std::round((resized.cols - w) / 2.0f);
  int top = std::round((resized.rows - h) / 2.0f);
  cv::Mat cropped = resized(cv::Rect(left, top, w, h));
  // img - mean
  cv::Mat float_mat;
  cropped.convertTo(float_mat, CV_32FC3, 1, -128);
  return float_mat;
}