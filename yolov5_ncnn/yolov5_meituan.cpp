// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>

//#define YOLOV5_V60 1 //YOLOv5 v6.0
#define YOLOV5_V62 1 //YOLOv5 v6.2 export  onnx model method https://github.com/shaoshengsong/yolov5_62_export_ncnn

#if YOLOV5_V60 || YOLOV5_V62
#define MAX_STRIDE 64
#else
#define MAX_STRIDE 32
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)
#endif //YOLOV5_V60    YOLOV5_V62

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);
                    if (confidence >= prob_threshold)
                    {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov5;

    yolov5.opt.use_vulkan_compute = true;
    // yolov5.opt.use_bf16_storage = true;

    // original pretrained model from https://github.com/ultralytics/yolov5
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    boost::filesystem::path file = boost::filesystem::canonical(__FILE__);
    // Get the parent directory
    boost::filesystem::path root = file.parent_path();
    std::cout << "Root Directory: " << root << std::endl;

    // Add ROOT to PATH if not already present
    bool pathExists = false;
    const char* pathVar = std::getenv("PATH");
    if (pathVar) {
        boost::filesystem::path pathEnv(pathVar);
        for (const boost::filesystem::path& path : pathEnv) {
            if (path == root) {
                pathExists = true;
                break;
            }
        }
    }

    if (!pathExists) {
        std::cout << "Adding ROOT to PATH" << std::endl;
        std::string pathStr = root.string();
        setenv("PATH", (std::string(getenv("PATH")) + ":" + pathStr).c_str(), 1);
    }

    // Construct file paths
    boost::filesystem::path modelparam = root / "meituan.param";
    boost::filesystem::path modelBin = root / "meituan.bin";

    // Print the file paths
    std::cout << "Model XML Path: " << modelparam << std::endl;
    std::cout << "Model BIN Path: " << modelBin << std::endl;


    if (yolov5.load_param(modelparam.c_str()))
        exit(-1);
    if (yolov5.load_model(modelBin.c_str()))
        exit(-1);

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;
    ex.extract("output", out0);
    ex.extract("354", out1);
    ex.extract("366", out2);

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out0, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out1, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out2, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }


    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

// static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
// {
//     // load ncnn model
//     ncnn::Net yolov5;

//     yolov5.opt.use_vulkan_compute = true;

//     yolov5.load_param("1.param");
//     yolov5.load_model("1.bin");

//     const int target_size = 640;
//     const float prob_threshold = 0.25f;
//     const float nms_threshold = 0.45f;

//     // load image, resize and pad to 640x640
//     const int img_w = bgr.cols;
//     const int img_h = bgr.rows;

//     // solve resize scale
//     int w = img_w;
//     int h = img_h;
//     float scale = 1.f;
//     if (w > h)
//     {
//         scale = (float)target_size / w;
//         w = target_size;
//         h = h * scale;
//     }
//     else
//     {
//         scale = (float)target_size / h;
//         h = target_size;
//         w = w * scale;
//     }

//     // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
//     ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

//     // pad to target_size rectangle
//     const int wpad = target_size - w;
//     const int hpad = target_size - h;
//     ncnn::Mat in_pad;
//     ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

//     // apply yolov5 pre process, that is to normalize 0~255 to 0~1
//     const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
//     in_pad.substract_mean_normalize(0, norm_vals);

//     // yolov5 model inference
//     ncnn::Extractor ex = yolov5.create_extractor();
//     ex.input("images", in_pad);
//     ncnn::Mat out;
//     ex.extract("/model.24/Transpose_2_output_0", out);

//     // the out blob would be a 2-dim tensor with w=85 h=25200
//     //
//     //        |cx|cy|bw|bh|box score(1)| per-class scores(80) |
//     //        +--+--+--+--+------------+----------------------+
//     //        |53|50|70|80|    0.11    |0.1 0.0 0.0 0.5 ......|
//     //   all /|  |  |  |  |      .     |           .          |
//     //  boxes |46|40|38|44|    0.95    |0.0 0.9 0.0 0.0 ......|
//     // (25200)|  |  |  |  |      .     |           .          |
//     //       \|  |  |  |  |      .     |           .          |
//     //        +--+--+--+--+------------+----------------------+
//     //

//     // enumerate all boxes
//     std::vector<Object> proposals;
//     for (int i = 0; i < out.h; i++)
//     {
//         const float* ptr = out.row(i);

//         const int num_class = 80;

//         const float cx = ptr[0];
//         const float cy = ptr[1];
//         const float bw = ptr[2];
//         const float bh = ptr[3];
//         const float box_score = ptr[4];
//         const float* class_scores = ptr + 5;

//         // find class index with the biggest class score among all classes
//         int class_index = 0;
//         float class_score = -FLT_MAX;
//         for (int j = 0; j < num_class; j++)
//         {
//             if (class_scores[j] > class_score)
//             {
//                 class_score = class_scores[j];
//                 class_index = j;
//             }
//         }

//         // combined score = box score * class score
//         float confidence = box_score * class_score;

//         // filter candidate boxes with combined score >= prob_threshold
//         if (confidence < prob_threshold)
//             continue;

//         // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
//         float x0 = cx - bw * 0.5f;
//         float y0 = cy - bh * 0.5f;
//         float x1 = cx + bw * 0.5f;
//         float y1 = cy + bh * 0.5f;

//         // collect candidates
//         Object obj;
//         obj.rect.x = x0;
//         obj.rect.y = y0;
//         obj.rect.width = x1 - x0;
//         obj.rect.height = y1 - y0;
//         obj.label = class_index;
//         obj.prob = confidence;

//         proposals.push_back(obj);
//     }

//     // sort all candidates by score from highest to lowest
//     qsort_descent_inplace(proposals);

//     // apply non max suppression
//     std::vector<int> picked;
//     nms_sorted_bboxes(proposals, picked, nms_threshold);

//     // collect final result after nms
//     const int count = picked.size();
//     objects.resize(count);
//     for (int i = 0; i < count; i++)
//     {
//         objects[i] = proposals[picked[i]];

//         // adjust offset to original unpadded
//         float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
//         float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
//         float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
//         float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

//         // clip
//         x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
//         y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
//         x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
//         y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

//         objects[i].rect.x = x0;
//         objects[i].rect.y = y0;
//         objects[i].rect.width = x1 - x0;
//         objects[i].rect.height = y1 - y0;
//     }

//     return 0;
// }

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"meituan"};
    
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolov5(m, objects);

    draw_objects(m, objects);

    return 0;
}

// int main(int argc, char** argv)
// {
//     // Check if the correct number of arguments is provided
//     if (argc != 1)
//     {
//         fprintf(stderr, "Usage: %s\n", argv[0]);
//         return -1;
//     }

//     // Open the default camera (camera index 0)
//     cv::VideoCapture cap(0);
//     if (!cap.isOpened())
//     {
//         fprintf(stderr, "Error: Could not open camera.\n");
//         return -1;
//     }
//     else
//     {
//         std::cout<<"Cam True"<<std::endl;
//     }

//     cv::Mat frame;

//     while (true)
//     {
//         // Capture frame-by-frame
//         cap >> frame;

//         // Check if the frame is empty
//         if (frame.empty())
//         {
//             fprintf(stderr, "Error: Failed to capture frame.\n");
//             break;
//         }

//         std::vector<Object> objects;
//         detect_yolov5(frame, objects);

//         draw_objects(frame, objects);

//         // Display the resulting frame
//         cv::imshow("YOLO Detection", frame);

//         // Break the loop if 'Esc' key is pressed
//         if (cv::waitKey(1) == 27)
//             break;
//     }

//     // Release the VideoCapture object
//     cap.release();

//     // Destroy all OpenCV windows
//     cv::destroyAllWindows();

//     return 0;
// }