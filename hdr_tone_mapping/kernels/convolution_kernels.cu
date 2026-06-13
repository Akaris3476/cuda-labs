#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>

namespace cuda_filter
{

// CUDA error checking
#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    // CUDA kernel for 2D convolution
    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            // Clamp the result to [0, 255]
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        // Copy kernel to CPU float array
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch kernel
        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        // Free host memory
        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        // Convert kernel to float array for faster access
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Process each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    // Apply kernel
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    // Clamp the result to [0, 255]
                    output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }




    __device__ inline float gpu_lerp(float a, float b, float t) {
        return a + t * (b - a);
    }

    __global__ void hdrGlobalKernel(const unsigned char *input, unsigned char *output,
                                    int width, int height, int channels,
                                    float exposure, float gamma, float saturation)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = (y * width + x) * channels;

        float b = (input[idx + 0] / 255.0f) * exposure;
        float g = (input[idx + 1] / 255.0f) * exposure;
        float r = (input[idx + 2] / 255.0f) * exposure;

        float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        float Y_mapped = Y / (1.0f + Y);

        float b_out = (Y > 0.0f) ? (b / Y) * Y_mapped : 0.0f;
        float g_out = (Y > 0.0f) ? (g / Y) * Y_mapped : 0.0f;
        float r_out = (Y > 0.0f) ? (r / Y) * Y_mapped : 0.0f;

        b_out = gpu_lerp(Y_mapped, b_out, saturation);
        g_out = gpu_lerp(Y_mapped, g_out, saturation);
        r_out = gpu_lerp(Y_mapped, r_out, saturation);

        b_out = powf(max(0.0f, b_out), 1.0f / gamma);
        g_out = powf(max(0.0f, g_out), 1.0f / gamma);
        r_out = powf(max(0.0f, r_out), 1.0f / gamma);

        output[idx + 0] = static_cast<unsigned char>(min(max(b_out * 255.0f, 0.0f), 255.0f));
        output[idx + 1] = static_cast<unsigned char>(min(max(g_out * 255.0f, 0.0f), 255.0f));
        output[idx + 2] = static_cast<unsigned char>(min(max(r_out * 255.0f, 0.0f), 255.0f));

        if (channels == 4) {
            output[idx + 3] = input[idx + 3];
        }
    }

    constexpr int TILE_SIZE = 16;
    constexpr int RAD = 2;
    constexpr int SH_SZ = (TILE_SIZE + 2 * RAD);

    __global__ void hdrLocalKernel(const unsigned char *input, unsigned char *output,
                                   int width, int height, int channels,
                                   float exposure, float gamma, float saturation)
    {
        __shared__ float s_Y[SH_SZ][SH_SZ];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * TILE_SIZE + tx;
        int y = blockIdx.y * TILE_SIZE + ty;

        int shared_y = ty;
        while (shared_y < SH_SZ) {
            int shared_x = tx;
            while (shared_x < SH_SZ) {
                int in_x = min(max(static_cast<int>(blockIdx.x * TILE_SIZE + shared_x - RAD), 0), width - 1);
                int in_y = min(max(static_cast<int>(blockIdx.y * TILE_SIZE + shared_y - RAD), 0), height - 1);

                int src_idx = (in_y * width + in_x) * channels;
                float b_sh = input[src_idx + 0] / 255.0f;
                float g_sh = input[src_idx + 1] / 255.0f;
                float r_sh = input[src_idx + 2] / 255.0f;

                s_Y[shared_y][shared_x] = 0.2126f * r_sh + 0.7152f * g_sh + 0.0722f * b_sh;

                shared_x += TILE_SIZE;
            }
            shared_y += TILE_SIZE;
        }
        __syncthreads();

        if (x >= width || y >= height) return;

        int idx = (y * width + x) * channels;
        float b = (input[idx + 0] / 255.0f) * exposure;
        float g = (input[idx + 1] / 255.0f) * exposure;
        float r = (input[idx + 2] / 255.0f) * exposure;
        float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;

        float Y_local_sum = 0.0f;
        for (int ky = 0; ky <= 2 * RAD; ++ky) {
            for (int kx = 0; kx <= 2 * RAD; ++kx) {
                Y_local_sum += s_Y[ty + ky][tx + kx];
            }
        }
        float Y_local_avg = Y_local_sum / ((2 * RAD + 1) * (2 * RAD + 1));

        float Y_mapped = Y / (1.0f + Y_local_avg);

        float b_out = (Y > 0.0f) ? (b / Y) * Y_mapped : 0.0f;
        float g_out = (Y > 0.0f) ? (g / Y) * Y_mapped : 0.0f;
        float r_out = (Y > 0.0f) ? (r / Y) * Y_mapped : 0.0f;

        b_out = gpu_lerp(Y_mapped, b_out, saturation);
        g_out = gpu_lerp(Y_mapped, g_out, saturation);
        r_out = gpu_lerp(Y_mapped, r_out, saturation);

        b_out = powf(max(0.0f, b_out), 1.0f / gamma);
        g_out = powf(max(0.0f, g_out), 1.0f / gamma);
        r_out = powf(max(0.0f, r_out), 1.0f / gamma);

        output[idx + 0] = static_cast<unsigned char>(min(max(b_out * 255.0f, 0.0f), 255.0f));
        output[idx + 1] = static_cast<unsigned char>(min(max(g_out * 255.0f, 0.0f), 255.0f));
        output[idx + 2] = static_cast<unsigned char>(min(max(r_out * 255.0f, 0.0f), 255.0f));

        if (channels == 4) {
            output[idx + 3] = input[idx + 3];
        }
    }
    void applyHDRToneMappingGPU(const cv::Mat &input, cv::Mat &output,
                                float exposure, float gamma, float saturation, bool use_local)
    {
        if (input.empty()) return;
        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();

        size_t imageSize = width * height * channels * sizeof(unsigned char);

        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        if (use_local) {
            hdrLocalKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, exposure, gamma, saturation);
        } else {
            hdrGlobalKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, exposure, gamma, saturation);
        }

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
    }

    void applyHDRToneMappingCPU(const cv::Mat &input, cv::Mat &output,
                                float exposure, float gamma, float saturation)
    {
        if (input.empty()) return;
        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();

        for (int y = 0; y < height; y++)
        {
            const unsigned char* in_row = input.ptr<unsigned char>(y);
            unsigned char* out_row = output.ptr<unsigned char>(y);

            for (int x = 0; x < width; x++)
            {
                int idx = x * channels;

                float b = (in_row[idx + 0] / 255.0f) * exposure;
                float g = (in_row[idx + 1] / 255.0f) * exposure;
                float r = (in_row[idx + 2] / 255.0f) * exposure;

                float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                float Y_mapped = Y / (1.0f + Y);

                float b_out = (Y > 0.0f) ? (b / Y) * Y_mapped : 0.0f;
                float g_out = (Y > 0.0f) ? (g / Y) * Y_mapped : 0.0f;
                float r_out = (Y > 0.0f) ? (r / Y) * Y_mapped : 0.0f;

                b_out = Y_mapped + saturation * (b_out - Y_mapped);
                g_out = Y_mapped + saturation * (g_out - Y_mapped);
                r_out = Y_mapped + saturation * (r_out - Y_mapped);

                b_out = std::pow(std::max(0.0f, b_out), 1.0f / gamma);
                g_out = std::pow(std::max(0.0f, g_out), 1.0f / gamma);
                r_out = std::pow(std::max(0.0f, r_out), 1.0f / gamma);

                out_row[idx + 0] = static_cast<unsigned char>(std::min(std::max(b_out * 255.0f, 0.0f), 255.0f));
                out_row[idx + 1] = static_cast<unsigned char>(std::min(std::max(g_out * 255.0f, 0.0f), 255.0f));
                out_row[idx + 2] = static_cast<unsigned char>(std::min(std::max(r_out * 255.0f, 0.0f), 255.0f));

                if (channels == 4) {
                    out_row[idx + 3] = in_row[idx + 3];
                }
            }
        }
    }



} // namespace cuda_filter
