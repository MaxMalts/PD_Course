#include <iostream>
#include <CosineVector.cuh>
#include <ScalarMulRunner.cuh>

__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    
    result[blockIdx.x] = 0;
	for (int i = index; i < numElements; i += stride) {
		atomicAdd(&result[blockIdx.x], vector1[i] * vector2[i]);
	}
}

__global__ void Reduce(float* array, float* res) {
    extern __shared__ float shared[];

    int threadId = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[threadId] = array[index];
    __syncthreads();
    
    for (int curSize = blockDim.x / 2; curSize > 0; curSize /= 2) {
        if (threadId < curSize) {
            shared[threadId] += shared[threadId + curSize];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        res[blockIdx.x] = shared[0];
    }
}

float CosineVector(int numElements, float* vector1, float* vector2, int blockSize) {
    float *d_vector1, *d_vector2, *d_mul, *d_res;
    
    int nBlocks = (numElements + blockSize - 1) / blockSize;

    cudaMalloc(&d_vector1, numElements * sizeof(float));
    cudaMalloc(&d_vector2, numElements * sizeof(float));
    cudaMalloc(&d_mul, nBlocks * blockSize * sizeof(float));
    cudaMalloc(&d_res, sizeof(float));
    cudaMemcpy(d_vector1, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    ScalarMulBlock<<<nBlocks, blockSize>>>(numElements, d_vector1, d_vector2, d_mul);
    cudaDeviceSynchronize();
    
    float nBlocks1 = (nBlocks + blockSize - 1) / blockSize;
    Reduce<<<nBlocks1, blockSize, blockSize * sizeof(float)>>>(d_mul, d_res);
    cudaDeviceSynchronize();
    
    float scalMul = 0;
    cudaMemcpy(&scalMul, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    ScalarMulBlock<<<nBlocks, blockSize>>>(numElements, d_vector1, d_vector1, d_mul);
    cudaDeviceSynchronize();
    
    Reduce<<<nBlocks1, blockSize, blockSize * sizeof(float)>>>(d_mul, d_res);
    cudaDeviceSynchronize();
    
    float len1sq = 0;
    cudaMemcpy(&len1sq, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    
    ScalarMulBlock<<<nBlocks, blockSize>>>(numElements, d_vector2, d_vector2, d_mul);
    cudaDeviceSynchronize();
    
    Reduce<<<nBlocks1, blockSize, blockSize * sizeof(float)>>>(d_mul, d_res);
    cudaDeviceSynchronize();
    
    float len2sq = 0;
    cudaMemcpy(&len2sq, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float run_time_ms = 0;
    cudaEventElapsedTime(&run_time_ms, start, stop);
    std::cout << run_time_ms << std::endl;

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_mul);
    cudaFree(d_res);
    return scalMul / (sqrt(len1sq) * sqrt(len2sq));
}