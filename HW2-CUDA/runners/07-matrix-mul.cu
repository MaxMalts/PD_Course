#include <iostream>
#include <MatrixMul.cuh>

int main() {
    int nElements = 0, blockSize = 0;
    std::cin >> nElements >> blockSize;
    int size = static_cast<int>(sqrt(nElements));
    
    float* a = (float*)malloc(size * size * sizeof(float));
    float* b = (float*)malloc(size * size * sizeof(float));
    float* res = (float*)malloc(size * size * sizeof(float));
    
    float *d_a, *d_b, *d_res;
    cudaMalloc(&d_a, size * size * sizeof(float));
    cudaMalloc(&d_b, size * size * sizeof(float));
    cudaMalloc(&d_res, size * size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[i * size + j] = 2.0f;
            b[i * size + j] = 3.0f;
        }
    }
    
    cudaMemcpy(d_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 nBlocks((size + blockSize - 1) / blockSize, (size + blockSize - 1) / blockSize);
    dim3 blockSizes(sqrt(blockSize), sqrt(blockSize));

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    MatrixMul<<<nBlocks, blockSizes>>>(size, size, size, d_a, d_b, d_res);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaDeviceSynchronize();
    cudaGetLastError();
    
    float run_time_ms = 0;
    cudaEventElapsedTime(&run_time_ms, start, stop);
    std::cout << run_time_ms << std::endl;
    
    cudaMemcpy(res, d_res, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    
    free(a);
    free(b);
    free(res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    return 0;
}