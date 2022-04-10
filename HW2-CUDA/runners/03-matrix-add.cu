#include <iostream>
#include <cmath>
#include "KernelMatrixAdd.cuh"

int main() {
    int nElements = 0, blockSize = 0;
    std::cin >> nElements >> blockSize;
    int size = static_cast<int>(sqrt(nElements));
    
    float* a = (float*)malloc(size * size * sizeof(float));
    float* b = (float*)malloc(size * size * sizeof(float));
    float* res = (float*)malloc(size * size * sizeof(float));
    
    float *d_a, *d_b, *d_res;
    size_t pitch = 0;
    cudaMallocPitch(&d_a, &pitch, size * sizeof(float), size);
    cudaMallocPitch(&d_b, &pitch, size * sizeof(float), size);
    cudaMallocPitch(&d_res, &pitch, size * sizeof(float), size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i * size + j] = 2.0f;
            b[i * size + j] = 3.0f;
        }
    }
    
    cudaMemcpy2D(d_a, pitch, a, size * sizeof(float), size * sizeof(float), size, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pitch, b, size * sizeof(float), size * sizeof(float), size, cudaMemcpyHostToDevice);

    int nBlocks = (size * pitch + blockSize - 1) / blockSize;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    KernelMatrixAdd<<<nBlocks, blockSize>>>(size, size, pitch, d_a, d_b, d_res);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaDeviceSynchronize();
    
    float run_time_ms = 0;
    cudaEventElapsedTime(&run_time_ms, start, stop);
    std::cout << run_time_ms << std::endl;
    
    cudaMemcpy2D(res, size * sizeof(float), d_res, pitch, size * sizeof(float), size, cudaMemcpyDeviceToHost);
    
    free(a);
    free(b);
    free(res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    return 0;
}