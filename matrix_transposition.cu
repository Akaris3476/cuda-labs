#include <iostream>

#define TILE 16

void initWith(float num, float *a, int n1, int n2)
{
    for (int row = 0; row < n1; ++row)
    {
        for (int column = 0; column < n2; column++)
        {
            a[row*n2 + column] = (float)column;

        }
    }
}

void CPUtranspose(float* a, float* b, int n1, int n2)
{
    for (int row = 0; row < n2; ++row)
    {
        for (int col = 0; col < n1; ++col)
        {
            b[row*n1 + col] = a[col*n2 + row];
        }
    }

}

__global__ void GPUtranspose(float* a, float* b, int n1, int n2)
{

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;


    if (row < n1 && col < n2)
    {
        b[col * n1 + row] = a[row * n2 + col];
    }
}

__global__ void GPUtransposeOptimized(float* a, float* b, int n1, int n2)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < n2 && y < n1)
    {
        tile[threadIdx.y][threadIdx.x] = a[y * n2 + x];
    }

    __syncthreads();

    int transposedX = blockIdx.y * TILE + threadIdx.x;
    int transposedY = blockIdx.x * TILE + threadIdx.y;

    if (transposedX < n1 && transposedY < n2)
    {
        b[transposedY * n1 + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}


int main()
{
    clock_t start1, end1;
    double overall_time;
    start1 = clock();


    int n1 = 2048;
    int n2 = 1024;

    size_t size = n1*n2*sizeof(float);

    float* a = (float*)malloc(size);
    float* b  = (float*)malloc(size);

    initWith(0, a , n1, n2);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);

    CPUtranspose(a, b, n1, n2);

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    float cpu_time = 0;
    cudaEventElapsedTime(&cpu_time, start2, stop2);
    printf("CPU time: %f ms\n", cpu_time);



    float *c_a, *c_b;

    cudaMallocManaged(&c_a, size);
    cudaMallocManaged(&c_b, size);

    initWith(0, c_a, n1, n2);


    dim3 threadsPerBlock(16, 16);

    dim3 numberOfBlocks(
        (n2 + 15) / 16,
        (n1 + 15) / 16
    );


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    GPUtranspose<<<numberOfBlocks, threadsPerBlock>>>(c_a, c_b, n1,n2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU time: %f ms\n", gpu_time);

    for (int i = 0; i < n1*n2; i++){
        if (b[i] != c_b[i])
        {
            std::cout << "Matrices don't match" << std::endl;
            break;
        }
    }


    float *c_a1, *c_b1;

    cudaMallocManaged(&c_a1, size);
    cudaMallocManaged(&c_b1, size);

    initWith(0, c_a1, n1, n2);



    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    cudaEventRecord(start3);

    GPUtransposeOptimized<<<numberOfBlocks, threadsPerBlock>>>(c_a1, c_b1, n1,n2);


    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);

    float gpu2_time = 0;
    cudaEventElapsedTime(&gpu2_time, start3, stop3);
    printf("Optimized GPU time: %f ms\n", gpu2_time);

    for (int i = 0; i < n1*n2; i++){
        if (b[i] != c_b1[i])
        {
            std::cout << "Matrices don't match 2" << std::endl;
            break;
        }
    }


    free(a);
    free(b);
    cudaFree(c_a);
    cudaFree(c_b);
    cudaFree(c_a1);
    cudaFree(c_b1);

    end1 = clock();

    overall_time = ((double)(end1 - start1)) / CLOCKS_PER_SEC;

    printf("Execution time: %f ms\n", overall_time * 1000);
    return 0;
}