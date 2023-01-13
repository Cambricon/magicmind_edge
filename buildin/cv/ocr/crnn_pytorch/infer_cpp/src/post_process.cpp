#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cmath>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

#include "post_process.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

static void log_softmax_cpu(float *logits, int max_len, int dict_len)
{
    for(int i=0; i<max_len; ++i)
    {
        double sum_exp = 0;
        
        float alpha = *max_element(logits+i*dict_len, logits+(i+1)*dict_len);
        for(int j=0; j<dict_len; ++j)
        {
            int idx = i*dict_len+j;            
            logits[idx] -= alpha;
            sum_exp += exp(logits[idx]);
        }

        for(int j=0; j<dict_len; ++j)
        {
            double log_sum_exp = log(sum_exp);
            int idx = i*dict_len+j;
            logits[idx] -= log_sum_exp;
        }
    }
}

static void softmax_cpu(float *logits, int max_len, int dict_len)
{
    for(int i=0; i<max_len; ++i)
    {
        double sum_exp = 0;
        
        float alpha = *max_element(logits+i*dict_len, logits+(i+1)*dict_len);
        for(int j=0; j<dict_len; ++j)
        {
            int idx = i*dict_len+j;
            logits[i*dict_len+j] = exp(logits[i*dict_len+j] - alpha);            
            sum_exp += logits[idx];
        }

        for(int j=0; j<dict_len; ++j)
        {        
            int idx = i*dict_len+j;
            logits[idx] /= sum_exp;
        }
    }
}


static string labels2string(vector<int> &label)
{
    char dictionary[38] = "-0123456789abcdefghijklmnopqrstuvwxyz";
    string result = "";
    for(int i=0; i<label.size(); ++i)
    {   
        // cout <<"label[i] "<< i<< ":" << label[i] << endl;
        result += dictionary[label[i]];
    }
    return result;
}

static string B_operation(vector<int> &labels, int blank=0)
{
    vector<int> new_labels;
    int previous = -1;
    for(int i=0; i<labels.size(); ++i)
    {
        if(labels[i] != previous)
        {
            previous = labels[i];
            if(labels[i] != 0)
            {
                new_labels.push_back(labels[i]);
                previous = labels[i];
            }                
        }
    }
    string result = labels2string(new_labels);
    return result;
}

static void argmax(float *emission_log_prob, vector<int> &labels, int max_len, int dict_len)
{
    for(int i=0; i<max_len; ++i)
    {
        int max_index = -1;
        int max_elem = -1;
        int index = i * dict_len;
        for(int j=0; j<dict_len; ++j)
        {
            if(emission_log_prob[index+j]>max_elem)
            {
                max_elem = emission_log_prob[index+j];
                max_index = j;
            }
        }
        labels.push_back(max_index);
    }
}

static string greedy_decode(float *emission_log_prob, int max_len, int dict_len)
{
    vector<int> labels;
    argmax(emission_log_prob, labels, max_len, dict_len);
    string result = B_operation(labels);
    return result;
}

string beam_search_decode(float *emission_log_prob, int max_len, int dict_len, int beam_size=10, float default_emission_threshold=0.01)
{
    float emission_threshold = log(default_emission_threshold);
}

string post_process(float *logits)
{    
    int max_len = 24;
    int dict_len = 37;
    log_softmax_cpu(logits, max_len, dict_len);    
    string result = greedy_decode(logits, max_len, dict_len);    
    return result;
}
