#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "preprocess.h"

//#define USE_RFB_320
#define USE_RFB_640
#define USE_FP16  // set USE_FP16 or USE_FP32
#define ULTAFACE_ONNX_FILE    "ultraface.onnx"
#define ULTAFACE_ENGINE_FILE  "ultraface.engine"

#define DEF_SCORE_THRESHOLD  0.8
#define DEF_IOU_THRESHOLD    0.2

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define EXIT_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "ultraface: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        exit(ret);                                                                       \
    } while(0)

#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 
#define MAX_WORKSPACE (1 << 20)    
// stuff we know about the network and the input/output blobs
#if defined(USE_RFB_320)
static const int INPUT_W = 320;
static const int INPUT_H = 240;
static const int OUTPUT_SIZE = 4420 * 4;
#elif defined(USE_RFB_640)
static const int INPUT_W = 640;
static const int INPUT_H = 480;
static const int OUTPUT_SIZE = 17640 * 4;
#else
# error Network input size is not defined!
#endif

// UltraFace variables
#define hard_nms 1
#define blending_nms 2

auto clip = [](float x, float y) {return (x < 0 ? 0 : (x > y ? y : x));};
const float center_variance = 0.1;
const float size_variance = 0.2;
std::vector<std::vector<float>> priors = {};
const std::vector<float> _strides = {8.0, 16.0, 32.0, 64.0};
const std::vector<std::vector<float>> _min_boxes = {{10.0f,  16.0f,  24.0f},
                                                    {32.0f,  48.0f},
                                                    {64.0f,  96.0f},
                                                    {128.0f, 192.0f, 256.0f}};

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

    float landmarks[10];
} FaceInfo;

using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;

/*
 *
 */
void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float iou_threshold, int type) {
    std::sort(input.begin(), input.end(), 
              [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
}

/*
 *
 */
void prepare_anchors(int iw, int ih) {
    int in_w = iw;
    int in_h = ih;
    auto w_h_list = {in_w, in_h};
    int num_featuremap = 4;
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    // TODO: see what should be values

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : _strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(_strides);
    }
    
    // Generate anchors
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : _min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1.f), clip(y_center, 1.f), clip(w, 1.f), clip(h, 1.f)});
                }
            }
        }
    }
}

/*
 *
 */
std::vector<FaceInfo> postprocess(float *buffer[], int width, int height) {
    // Convert bounding boxes
    float *score_value = buffer[0];
    float *bbox_value =  buffer[1];
    std::vector<FaceInfo> bbox_collection;

    for (int i = 0; i < priors.size(); i++) {
        float score = score_value[2 * i + 1];
        if (score_value[2 * i + 1] > DEF_SCORE_THRESHOLD) {
            FaceInfo rects = {0};
            // Calculate rect coordinates
            float x_center = bbox_value[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = bbox_value[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(bbox_value[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(bbox_value[i * 4 + 3] * size_variance) * priors[i][3];

            // Add bbox
            rects.x1 = clip(x_center - w / 2.0, 1) * width;
            rects.y1 = clip(y_center - h / 2.0, 1) * height;
            rects.x2 = clip(x_center + w / 2.0, 1) * width;
            rects.y2 = clip(y_center + h / 2.0, 1) * height;
            rects.score = clip(score_value[2 * i + 1], 1);
            bbox_collection.push_back(rects);
        }
    }

    std::vector<FaceInfo> result_collection;
    nms(bbox_collection, result_collection, DEF_IOU_THRESHOLD, blending_nms);

    return result_collection;
}

/*
 *
 */
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config) {
    uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);
    IParser* parser = createParser(*network, gLogger);

    if (!parser->parseFromFile("../"ULTAFACE_ONNX_FILE, static_cast<int32_t>(ILogger::Severity::kWARNING))) {
        EXIT_AND_LOG(-1, ERROR, "Fail to parse ONNX");
    }
    /* we create the engine */
    printf("ONNX model parse done\n");
    
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{1, 3, INPUT_H, INPUT_W});
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{1, 3, INPUT_H, INPUT_W});
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{1, 3, INPUT_H, INPUT_W});
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif    
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(MAX_WORKSPACE);
    config->addOptimizationProfile(profile);

    printf("Building the engine...\n");

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);//builder->buildCudaEngine(*network);
    if (!engine) {
        printf("Build engine failed!\n");
        EXIT_AND_LOG(-1, ERROR, "Unable to create engine. Perhaps run this executable as root");
    }

    printf("Build done!\n");

    parser->destroy();
    network->destroy();

    return engine;
}

/*
 *
 */
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

/*
 *
 */
void doInference(IExecutionContext& context,  cudaStream_t& stream, void **buffers, 
                 float *output[], int batchSize, std::vector<int>& _output_indexes) {
    
    context.enqueue(batchSize, buffers, stream, nullptr);
    
    for (int i = 0; i < _output_indexes.size(); i++) {
        CHECK(cudaMemcpyAsync(output[i], buffers[_output_indexes[i]], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}

/*
 *
 */
std::size_t get_size_by_dim(const nvinfer1::Dims& dims) {
    std::size_t size = 1;
    for (std::size_t i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

/*
 *
 */
int main(int argc, char** argv) {
    int batchSize = 1;

    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./ultraface -s   // serialize model to plan file" << std::endl;
        std::cerr << "./ultraface -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(batchSize, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("./"ULTAFACE_ENGINE_FILE, std::ios::binary);
        if (!p) {
            printf("Could not open plain output file\n");
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        printf("Engine created successfully!\n");
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("./"ULTAFACE_ENGINE_FILE, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        } else {
            printf("Failed to load engine file!\n");
            return -1;
        }
    } else {
        return -1;
    }

    //**********************************
    float *pdata;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    std::vector<nvinfer1::Dims> _input_dimensions;
    std::vector<nvinfer1::Dims> _output_dimensions;
    std::vector<int> _input_indexes;
    std::vector<int> _output_indexes;

    // input and output bufuffers
    void *buffers[engine->getNbBindings()];
    
    // prepare engine
    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i)) {
            _input_dimensions.emplace_back(engine->getBindingDimensions(i));
            _input_indexes.emplace_back(i);
        } else {
            _output_dimensions.emplace_back(engine->getBindingDimensions(i));
            _output_indexes.emplace_back(i);
        }
    }
    // 1 input and 2 outputs for ultraface
    assert(_input_indexes.size() == 1);
    assert(_output_indexes.size() == 2);

    float *prob[_output_indexes.size()];

    for (int i = 0;i < _output_indexes.size();i++) {
        CHECK(cudaMallocHost((void **)&prob[i], batchSize * OUTPUT_SIZE * sizeof(float)));
    }

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[_input_indexes[0]], batchSize * get_size_by_dim(_input_dimensions[0]) * sizeof(float)));
    // Allocate GPU buffers for output
    for (int i = 0; i < _output_indexes.size(); i++) {
        CHECK(cudaMalloc(&buffers[_output_indexes[i]], batchSize * OUTPUT_SIZE * sizeof(float)));
    }

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare UltraFace detector
    prepare_anchors(INPUT_W, INPUT_H);

    //************* workflow *********************************
    int ROI_number = 0;
    cv::Mat frame_in = cv::imread("../26.jpg");

    auto  t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        cv::Mat frame = frame_in.clone();
        float* buffer_idx = (float*)buffers[_input_indexes[0]];
        size_t  size_image = frame.cols * frame.rows * 3;
        // preprocess the image
#ifndef USE_CV_BLOB        
        memcpy(img_host, frame.data, size_image);
        CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device, frame.cols, frame.rows, buffer_idx, INPUT_W, INPUT_H, stream);
#else 
        cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0 / 128, cv::Size(INPUT_W, INPUT_H), cv::Scalar(127, 127, 127), true);
        CHECK(cudaMemcpyAsync(buffer_idx, inputBlob.data, batchSize * get_size_by_dim(_input_dimensions[0]) * sizeof(float), cudaMemcpyHostToDevice, stream));
#endif        
        // inferrence
        doInference(*context, stream, (void **)buffers, prob, batchSize, _output_indexes);
        // postprocess output
        std::vector<FaceInfo> face_boxes = postprocess(prob, frame.cols, frame.rows);
        // draw boxes
        ROI_number = 0;
        for (auto box : face_boxes) {
            int x1 = box.x1;
            int y1 = box.y1;
            int x2 = box.x2;
            int y2 = box.y2;
            int w = x2 - x1;
            int h = y2 - y1;

            ROI_number++;

            if (i == 0) {
                cv::Rect roi = cv::Rect(cv::Point(x1, y1), cv::Point(w, h));
                rectangle(frame, cv::Rect(x1, y1, w, h), cv::Scalar(255, 0, 0), 1);
            }
            
        }
        if (i == 0) {
            printf("Regions found: %d\n", ROI_number);
            cv::imwrite("./ultraface.jpg", frame);
        }
        //
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>
                                                    ((t_end - t_start) / 100.0);
    std::cout << "Time per inference: " << timeTaken.count() << " ms" << std::endl
              << "FPS: " << 1000.0 / (timeTaken.count()) << std::endl;

    //************ free and destroy the engine *******************
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[_input_indexes[0]]));
    for (int i = 0; i < _output_indexes.size(); i++) {
        CHECK(cudaFree(buffers[_output_indexes[i]]));
    }
    // Destroy the engine
    for (int i = 0;i < _output_indexes.size();i++) {
        CHECK(cudaFreeHost(prob[i]));
    }
    context->destroy();
    engine->destroy();
    runtime->destroy();
}
