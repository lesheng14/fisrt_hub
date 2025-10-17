#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================



// ==================== IF神经元核函数 - 硬重置版本 ====================
__global__ void if_neuron_hard_reset_kernel(
    float* input, float* membrane_potential,
    const float v_threshold, const float v_reset, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v_old = membrane_potential[idx];
        float v_new = v_old + input[idx];
        float spike = (v_new >= v_threshold) ? 1.0f : 0.0f;
        float v_next = (spike > 0.5f) ? v_reset : v_new;
        membrane_potential[idx] = v_next;
        input[idx] = spike; // overwrite input with spike
    }
}

static inline void if_neuron(
    float* d_input, float* d_membrane_potential,
    float v_threshold, float v_reset, int size,
    cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    if_neuron_hard_reset_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_membrane_potential, v_threshold, v_reset, size);
}


// ==================== 2D卷积核函数 ====================
__global__ void conv2d_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int padding)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;

    if (oc < out_channels && oh < out_height && ow < out_width) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        int input_idx = ic * (in_height * in_width) + ih * in_width + iw;
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size)
                                       + ic * (kernel_size * kernel_size)
                                       + kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        if (bias != nullptr) { sum += bias[oc]; }
        int output_idx = oc * (out_height * out_width) + oh * out_width + ow;
        output[output_idx] = sum;
    }
}

static inline void conv2d(
    const float* d_input, const float* d_weight, const float* d_bias,
    float* d_output,
    int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding,
    cudaStream_t stream = 0)
{
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    dim3 blockSize(4, 4, 4);
    dim3 gridSize(
        (out_channels + blockSize.x - 1) / blockSize.x,
        (out_height  + blockSize.y - 1) / blockSize.y,
        (out_width   + blockSize.z - 1) / blockSize.z
    );

    conv2d_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_weight, d_bias, d_output,
        in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding
    );
}


// ==================== 最大池化核函数 ====================
__global__ void max_pool2d_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int channels, int in_height, int in_width,
    int pool_size, int stride)
{
    int c  = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;

    int out_height = (in_height - pool_size) / stride + 1;
    int out_width  = (in_width  - pool_size) / stride + 1;

    if (c < channels && oh < out_height && ow < out_width) {
        int out_idx = c * (out_height * out_width) + oh * out_width + ow;
        float max_val = -1e9f;
        for (int ph = 0; ph < pool_size; ++ph) {
            for (int pw = 0; pw < pool_size; ++pw) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                if (ih < in_height && iw < in_width) {
                    int in_idx = c * (in_height * in_width) + ih * in_width + iw;
                    float v = input[in_idx];
                    max_val = v > max_val ? v : max_val;
                }
            }
        }
        output[out_idx] = max_val;
    }
}

static inline void max_pool2d(
    const float* d_input, float* d_output,
    int channels, int in_height, int in_width,
    int pool_size, int stride,
    cudaStream_t stream = 0)
{
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width  = (in_width  - pool_size) / stride + 1;

    dim3 blockSize(8, 4, 4);
    dim3 gridSize(
        (channels   + blockSize.x - 1) / blockSize.x,
        (out_height + blockSize.y - 1) / blockSize.y,
        (out_width  + blockSize.z - 1) / blockSize.z
    );

    max_pool2d_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_output,
        channels, in_height, in_width,
        pool_size, stride
    );
}


// ==================== 全连接层核函数 ====================
__global__ void linear_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output,
    int in_features, int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < out_features) {
        float sum = 0.0f;
        int base = out_idx * in_features;
        for (int in_idx = 0; in_idx < in_features; ++in_idx) {
            sum += input[in_idx] * weight[base + in_idx];
        }
        if (bias != nullptr) { sum += bias[out_idx]; }
        output[out_idx] = sum;
    }
}

static inline void linear(
    const float* d_input, const float* d_weight, const float* d_bias,
    float* d_output,
    int in_features, int out_features,
    cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = (out_features + blockSize - 1) / blockSize;
    linear_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_weight, d_bias, d_output,
        in_features, out_features
    );
}


// ==================== 向量操作核函数 ====================
__global__ void vector_add_kernel(const float* a, const float* b, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { output[idx] = a[idx] + b[idx]; }
}

static inline void vector_add(const float* d_a, const float* d_b, float* d_output, int size, cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    vector_add_kernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_output, size);
}

__global__ void vector_scale_kernel(const float* input, float scalar, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { output[idx] = input[idx] * scalar; }
}

static inline void vector_scale(const float* d_input, float scalar, float* d_output, int size, cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    vector_scale_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, scalar, d_output, size);
}


// ==================== 展平操作核函数 ====================
__global__ void flatten_kernel(const float* input, float* output, int channels, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = channels * height * width;
    if (idx < total_size) {
        int c  = idx / (height * width);
        int hw = idx % (height * width);
        int h  = hw / width;
        int w  = hw % width;
        int input_idx = c * (height * width) + h * width + w;
        output[idx] = input[input_idx];
    }
}

static inline void flatten(const float* d_input, float* d_output, int channels, int height, int width, cudaStream_t stream = 0)
{
    int total_size = channels * height * width;
    int blockSize = 256;
    int gridSize = (total_size + blockSize - 1) / blockSize;
    flatten_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, channels, height, width);
}

// ==================== 设备端 Argmax 内核 ====================
__global__ void argmax_kernel(const float* __restrict__ input, int size, int* __restrict__ out_index)
{
    // 数量很小（10 类），单线程实现即可，避免额外同步与共享内存开销
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int best_idx = 0;
        float best_val = input[0];
        for (int i = 1; i < size; ++i) {
            float v = input[i];
            if (v > best_val) { best_val = v; best_idx = i; }
        }
        *out_index = best_idx;
    }
}

static inline void argmax_device(const float* d_input, int size, int* d_out_index, cudaStream_t stream = 0)
{
    argmax_kernel<<<1, 1, 0, stream>>>(d_input, size, d_out_index);
}


std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b
    // YOU CAN ADD MORE PARAMETERS HERE!!!
    )
{
    std::vector<int> predictions;
    const int num_images = static_cast<int>(images.size());
    predictions.reserve(num_images);

    // SNN-specific parameter, must match training
    const int T = 8;

    // Network dimensions (FashionMNIST: 1x28x28)
    const int in_channels = 1;
    const int in_height   = 28;
    const int in_width    = 28;

    // conv1: 1 -> 6, k=5, s=1, p=0
    const int conv1_out_channels = 6;
    const int conv1_kernel = 5, conv1_stride = 1, conv1_pad = 0;
    const int conv1_out_height = (in_height + 2 * conv1_pad - conv1_kernel) / conv1_stride + 1; // 24
    const int conv1_out_width  = (in_width  + 2 * conv1_pad - conv1_kernel) / conv1_stride + 1; // 24
    const int conv1_out_elems  = conv1_out_channels * conv1_out_height * conv1_out_width;       // 6*24*24=3456

    // pool1: k=2, s=2
    const int pool_size = 2, pool_stride = 2;
    const int pool1_out_height = (conv1_out_height - pool_size) / pool_stride + 1; // 12
    const int pool1_out_width  = (conv1_out_width  - pool_size) / pool_stride + 1; // 12
    const int pool1_out_elems  = conv1_out_channels * pool1_out_height * pool1_out_width; // 6*12*12=864

    // conv2: 6 -> 16, k=5, s=1, p=0
    const int conv2_in_channels  = conv1_out_channels;
    const int conv2_out_channels = 16;
    const int conv2_kernel = 5, conv2_stride = 1, conv2_pad = 0;
    const int conv2_out_height = (pool1_out_height + 2 * conv2_pad - conv2_kernel) / conv2_stride + 1; // 8
    const int conv2_out_width  = (pool1_out_width  + 2 * conv2_pad - conv2_kernel) / conv2_stride + 1; // 8
    const int conv2_out_elems  = conv2_out_channels * conv2_out_height * conv2_out_width; // 16*8*8=1024

    // pool2: k=2, s=2
    const int pool2_out_height = (conv2_out_height - pool_size) / pool_stride + 1; // 4
    const int pool2_out_width  = (conv2_out_width  - pool_size) / pool_stride + 1; // 4
    const int pool2_out_elems  = conv2_out_channels * pool2_out_height * pool2_out_width; // 16*4*4=256

    // FC layers
    const int fc1_in_features  = pool2_out_elems; // 256
    const int fc1_out_features = 120;
    const int fc2_in_features  = fc1_out_features; // 120
    const int fc2_out_features = 84;
    const int fc3_in_features  = fc2_out_features; // 84
    const int fc3_out_features = 10;

    // IF neuron parameters (hard reset)
    const float v_threshold = 1.0f;
    const float v_reset     = 0.0f;

    // --- Allocate reusable device buffers (once) ---
    float *d_image = nullptr;
    float *d_conv1_out = nullptr, *d_pool1 = nullptr;
    float *d_conv2_out = nullptr, *d_pool2 = nullptr, *d_flatten = nullptr;
    float *d_fc1_out = nullptr, *d_fc2_out = nullptr, *d_fc3_out = nullptr;

    // Membrane potentials for IF nodes
    float *d_m1 = nullptr, *d_m2 = nullptr, *d_m3 = nullptr, *d_m4 = nullptr;
    // Accumulated logits and argmax index on device
    float *d_logits_sum = nullptr;
    int   *d_pred_index = nullptr;

    checkCudaErrors(cudaMalloc(&d_image,     in_channels * in_height * in_width * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_out, conv1_out_elems * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pool1,     pool1_out_elems * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_out, conv2_out_elems * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pool2,     pool2_out_elems * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_flatten,   fc1_in_features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_out,   fc1_out_features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_out,   fc2_out_features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_out,   fc3_out_features * sizeof(float)));

    checkCudaErrors(cudaMalloc(&d_m1, conv1_out_elems * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_m2, conv2_out_elems * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_m3, fc1_out_features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_m4, fc2_out_features * sizeof(float)));
    // Allocate device buffers for logits accumulation and argmax
    checkCudaErrors(cudaMalloc(&d_logits_sum, fc3_out_features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pred_index, sizeof(int)));

    // --- Loop over images ---
    for (int i = 0; i < num_images; ++i) {
        // Copy input image to device
        checkCudaErrors(cudaMemcpy(
            d_image, images[i].data(),
            in_channels * in_height * in_width * sizeof(float),
            cudaMemcpyHostToDevice));

        // Reset membrane potentials for this sample
        checkCudaErrors(cudaMemset(d_m1, 0, conv1_out_elems   * sizeof(float)));
        checkCudaErrors(cudaMemset(d_m2, 0, conv2_out_elems   * sizeof(float)));
        checkCudaErrors(cudaMemset(d_m3, 0, fc1_out_features  * sizeof(float)));
        checkCudaErrors(cudaMemset(d_m4, 0, fc2_out_features  * sizeof(float)));

        // Accumulate logits across T steps on device
        checkCudaErrors(cudaMemset(d_logits_sum, 0, fc3_out_features * sizeof(float)));

        for (int t = 0; t < T; ++t) {
            // conv1 -> IF1 -> pool1
            conv2d(
                d_image, d_conv1_w, d_conv1_b, d_conv1_out,
                in_channels, in_height, in_width,
                conv1_out_channels, conv1_kernel, conv1_stride, conv1_pad
            );
            if_neuron(d_conv1_out, d_m1, v_threshold, v_reset, conv1_out_elems);
            max_pool2d(
                d_conv1_out, d_pool1,
                conv1_out_channels, conv1_out_height, conv1_out_width,
                pool_size, pool_stride
            );

            // conv2 -> IF2 -> pool2
            conv2d(
                d_pool1, d_conv2_w, d_conv2_b, d_conv2_out,
                conv2_in_channels, pool1_out_height, pool1_out_width,
                conv2_out_channels, conv2_kernel, conv2_stride, conv2_pad
            );
            if_neuron(d_conv2_out, d_m2, v_threshold, v_reset, conv2_out_elems);
            max_pool2d(
                d_conv2_out, d_pool2,
                conv2_out_channels, conv2_out_height, conv2_out_width,
                pool_size, pool_stride
            );

            // flatten
            flatten(d_pool2, d_flatten, conv2_out_channels, pool2_out_height, pool2_out_width);

            // fc1 -> IF3
            linear(d_flatten, d_fc1_w, d_fc1_b, d_fc1_out, fc1_in_features, fc1_out_features);
            if_neuron(d_fc1_out, d_m3, v_threshold, v_reset, fc1_out_features);

            // fc2 -> IF4
            linear(d_fc1_out, d_fc2_w, d_fc2_b, d_fc2_out, fc2_in_features, fc2_out_features);
            if_neuron(d_fc2_out, d_m4, v_threshold, v_reset, fc2_out_features);

            // fc3
            linear(d_fc2_out, d_fc3_w, d_fc3_b, d_fc3_out, fc3_in_features, fc3_out_features);
            // Accumulate logits on device to avoid host-device transfers per timestep
            vector_add(d_logits_sum, d_fc3_out, d_logits_sum, fc3_out_features);
        }

        // select argmax on device and copy back a single int
        argmax_device(d_logits_sum, fc3_out_features, d_pred_index);
        int pred = 0;
        checkCudaErrors(cudaMemcpy(&pred, d_pred_index, sizeof(int), cudaMemcpyDeviceToHost));
        predictions.push_back(pred);
    }

    // free device buffers
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_conv1_out));
    checkCudaErrors(cudaFree(d_pool1));
    checkCudaErrors(cudaFree(d_conv2_out));
    checkCudaErrors(cudaFree(d_pool2));
    checkCudaErrors(cudaFree(d_flatten));
    checkCudaErrors(cudaFree(d_fc1_out));
    checkCudaErrors(cudaFree(d_fc2_out));
    checkCudaErrors(cudaFree(d_fc3_out));
    checkCudaErrors(cudaFree(d_m1));
    checkCudaErrors(cudaFree(d_m2));
    checkCudaErrors(cudaFree(d_m3));
    checkCudaErrors(cudaFree(d_m4));
    checkCudaErrors(cudaFree(d_logits_sum));
    checkCudaErrors(cudaFree(d_pred_index));

    return predictions;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
    std::string dir = argv[1];

    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");

    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
        // YOU CAN ADD MORE PARAMETERS HERE!!!
        );

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));

    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();

    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;

    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================
