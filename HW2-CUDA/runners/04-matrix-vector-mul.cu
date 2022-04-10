#include <iostream>
#include <cmath>
#include "MatrixVectorMul.cuh"

int main() {
    int nElements = 0, blockSize = 0;
    std::cin >> nElements >> blockSize;
    int size = static_cast<int>(sqrt(nElements));
    
    float* a = (float*)malloc(size * size * sizeof(float));
    float* v = (float*)malloc(size * sizeof(float));
    float* res = (float*)malloc(size * sizeof(float));
    
    float *d_a, *d_v, *d_res;
    size_t pitch = 0;
    cudaMallocPitch(&d_a, &pitch, size * sizeof(float), size);
    cudaMalloc(&d_v, size * sizeof(float));
    cudaMalloc(&d_res, size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i * size + j] = 2.0f;
        }
    }
    for (int i = 0; i < size; ++i) {
        v[i] = 3.0f;
    }
    
    cudaMemcpy2D(d_a, pitch, a, size * sizeof(float), size * sizeof(float), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice);

    int nBlocks = (size + blockSize - 1) / blockSize;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    MatrixVectorMul<<<nBlocks, blockSize>>>(size, size, pitch, d_a, d_v, d_res);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaDeviceSynchronize();
    
    float run_time_ms = 0;
    cudaEventElapsedTime(&run_time_ms, start, stop);
    std::cout << run_time_ms << std::endl;
    
    cudaMemcpy(res, d_res, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    free(a);
    free(v);
    free(res);
    cudaFree(d_a);
    cudaFree(d_v);
    cudaFree(d_res);
    return 0;
}