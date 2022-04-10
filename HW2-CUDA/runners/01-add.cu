#include <iostream>
#include <cmath>
#include "KernelAdd.cuh"

int main() {
    int nElements = 0, blockSize = 0;
    std::cin >> nElements >> blockSize;
    
    float *x, *y, *res;
    
    cudaMallocManaged(&x, nElements * sizeof(float));
    cudaMallocManaged(&y, nElements * sizeof(float));
    cudaMallocManaged(&res, nElements * sizeof(float));

    for (int i = 0; i < nElements; ++i) {
        x[i] = 2.0f;
        y[i] = 3.0f;
    }

    int nBlocks = (nElements + blockSize - 1) / blockSize;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    KernelAdd<<<nBlocks, blockSize>>>(nElements, x, y, res);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();	
    
    float run_time_ms = 0;
    cudaEventElapsedTime(&run_time_ms, start, stop);
    std::cout << run_time_ms << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(res);
    return 0;
}