#include <iostream>
#include <vector>
#include <chrono>


void convolveCPU(const float* input, float* output, int width, int height,
                 const float* kernel, int kSize) {

    int radius = kSize / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            float sum = 0.0f;

            for (int ky = 0; ky < kSize; ky++) {
                for (int kx = 0; kx < kSize; kx++) {

                    int ix = x + (kx - radius);
                    int iy = y + (ky - radius);


                    ix = std::max(0, std::min(width - 1, ix));
                    iy = std::max(0, std::min(height - 1, iy));

                    sum += input[iy * width + ix] * kernel[ky * kSize + kx];
                }
            }

            output[y * width + x] = sum;
        }
    }
}

__global__ void convolveNaive(const float* input, float* output, int width, int height,
                              const float* kernel, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < width && y < height) {
        int radius = kSize / 2;
        float sum = 0.0f;

        for (int ky = 0; ky < kSize; ky++) {
            for (int kx = 0; kx < kSize; kx++) {

                int ix = x + (kx - radius);
                int iy = y + (ky - radius);

                if (ix < 0) ix = 0;
                if (ix >= width) ix = width - 1;
                if (iy < 0) iy = 0;
                if (iy >= height) iy = height - 1;

                sum += input[iy * width + ix] * kernel[ky * kSize + kx];
            }
        }

        output[y * width + x] = sum;
    }
}

constexpr int K_SIZE = 7;
constexpr int TILE_SIZE  = 16;
constexpr int SHARED_SIZE = (TILE_SIZE + K_SIZE - 1);
__constant__ float d_const_kernel[K_SIZE*K_SIZE];

__global__ void convolveShared(const float* input, float* output, int width, int height, int kSize) {
    __shared__ float s_data[SHARED_SIZE][SHARED_SIZE];

    int radius = kSize / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;


    int shared_y = ty;
    while (shared_y < SHARED_SIZE) {
        int shared_x = tx;
        while (shared_x < SHARED_SIZE) {

            int input_x = blockIdx.x * TILE_SIZE + shared_x - radius;
            int input_y = blockIdx.y * TILE_SIZE + shared_y - radius;

            input_x = max(0, min(width - 1, input_x));
            input_y = max(0, min(height - 1, input_y));

            s_data[shared_y][shared_x] = input[input_y * width + input_x];

            shared_x += TILE_SIZE;
        }
        shared_y += TILE_SIZE;
    }

    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < kSize; ky++) {
            for (int kx = 0; kx < kSize; kx++) {
                sum += s_data[ty + ky][tx + kx] * d_const_kernel[ky * kSize + kx];
            }
        }
        output[y * width + x] = sum;
    }
}


__constant__ float d_const_kernel1D[K_SIZE];

__global__ void convolveRows(const float* input, float* output, int width, int height, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int radius = kSize / 2;
        float sum = 0.0f;
        for (int i = 0; i < kSize; i++) {
            int ix = x + (i - radius);
            ix = max(0, min(width - 1, ix));
            sum += input[y * width + ix] * d_const_kernel1D[i];
        }
        output[y * width + x] = sum;
    }
}

__global__ void convolveCols(const float* input, float* output, int width, int height, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int radius = kSize / 2;
        float sum = 0.0f;
        for (int i = 0; i < kSize; i++) {
            int iy = y + (i - radius);
            iy = max(0, min(height - 1, iy));
            sum += input[iy * width + x] * d_const_kernel1D[i];
        }
        output[y * width + x] = sum;
    }
}




int main() {
    const int W = 4096;
    const int H = 4096;


    std::vector<float> input(W * H, 3.0f);
    std::vector<float> output(W * H, 0.0f);
    std::vector<float> output_gpu_naive(W * H, 0.0f);
    std::vector<float> output_gpu(W * H, 0.0f);
    std::vector<float> output_gpu_separable(W * H, 0.0f);

    std::vector<float> kernel(K_SIZE * K_SIZE);
    std::vector<float> kernel1D(K_SIZE, 1.0f / K_SIZE);

    float weight = 1.0f / (K_SIZE * K_SIZE);
    for (int i = 0; i < K_SIZE * K_SIZE; i++) {
        kernel[i] = weight;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();

    convolveCPU(input.data(), output.data(), W, H, kernel.data(), K_SIZE);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_cpu - start_cpu;

    std::cout << "CPU Time: " << duration.count() << " ms" << std::endl;


    int size = W * H * sizeof(float);
    int kBytes = K_SIZE * K_SIZE * sizeof(float);
    int kBytes1D = K_SIZE * sizeof(float);

    float *d_input, *d_output, *d_kernel, *d_temp;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_kernel, kBytes);
    cudaMalloc(&d_temp, size);

    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x,
                  (H + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop, start1, stop1, start2, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    convolveNaive<<<gridSize, blockSize>>>(d_input, d_output, W, H, d_kernel, K_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU naive Time: " << gpu_time << " ms" << std::endl;

    cudaMemcpy(output_gpu_naive.data(), d_output, size, cudaMemcpyDeviceToHost);


    cudaMemcpyToSymbol(d_const_kernel, kernel.data(), kBytes);

    dim3 blockSize1(TILE_SIZE, TILE_SIZE);
    dim3 gridSize1((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);

    convolveShared<<<gridSize1, blockSize1>>>(d_input, d_output, W, H, K_SIZE);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start1, stop1);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    cudaMemcpy(output_gpu.data(), d_output, size, cudaMemcpyDeviceToHost);



    cudaMemcpyToSymbol(d_const_kernel1D, kernel1D.data(), kBytes1D);

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    convolveRows<<<gridSize, blockSize>>>(d_input, d_temp, W, H, K_SIZE);
    convolveCols<<<gridSize, blockSize>>>(d_temp, d_output, W, H, K_SIZE);

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start2, stop2);
    std::cout << "GPU Separable filters Time: " << gpu_time << " ms" << std::endl;

    cudaMemcpy(output_gpu_separable.data(), d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < W * H; i++) {
        if (output[i] != output_gpu_naive[i]) {
            std::cout << "NOT CORRECT" << std::endl;
            break;
        }
    }

    for (int i = 0; i < W * H; i++) {
        if (output[i] != output_gpu[i]) {
            std::cout << "NOT CORRECT 2" << std::endl;
            break;
        }
    }

    for (int i = 0; i < W * H; i++) {
        if (std::abs(output[i] - output_gpu_separable[i]) > 0.001f) {
            std::cout << "NOT CORRECT 2" << std::endl;
            break;
        }
    }


    return 0;
}