#include <iostream>
#include <ScalarMul.cuh>
#include <ScalarMulRunner.cuh>

__global__ void KernelMul(int numElements, float* x, float* y, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < numElements; i += stride) {
		result[i] = x[i] * y[i];
	}
}

__global__ void Reduce(float* array, float* res) {
    extern __shared__ float shared[];

    int threadId = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
//printf("%d\n", index);
    shared[threadId] = array[index];
    __syncthreads();
    
    for (int curSize = 1; curSize < blockDim.x; curSize *= 2) {
        int i = 2 * curSize * threadId;

        if (i < blockDim.x) {
            shared[i] += shared[i + curSize];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        res[blockIdx.x] = shared[0];
    }
}

float ScalarMulTwoReductions(int numElements, float* vector1, float* vector2, int blockSize) {
    float *d_vector1, *d_vector2, *d_mul, *d_reduce, *d_res;

    int nBlocks = (numElements + blockSize - 1) / blockSize;
    
    cudaMalloc(&d_vector1, numElements * sizeof(float));
    cudaMalloc(&d_vector2, numElements * sizeof(float));
    cudaMalloc(&d_mul, numElements * sizeof(float));
    cudaMalloc(&d_reduce, nBlocks * blockSize * sizeof(float));
    cudaMalloc(&d_res, sizeof(float));

    cudaMemcpy(d_vector1, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    KernelMul<<<nBlocks, blockSize>>>(numElements, d_vector1, d_vector2, d_mul);
    cudaDeviceSynchronize();
    
    Reduce<<<nBlocks, blockSize, blockSize * sizeof(float)>>>(d_mul, d_reduce);
    cudaDeviceSynchronize();
    
    nBlocks = (nBlocks + blockSize - 1) / blockSize;
    Reduce<<<nBlocks, blockSize, blockSize * sizeof(float)>>>(d_reduce, d_res);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float run_time_ms = 0;
    cudaEventElapsedTime(&run_time_ms, start, stop);
    std::cout << run_time_ms << std::endl;
    
    float res = 0;
    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_mul);
    cudaFree(d_reduce);
    cudaFree(d_res);
    return res;
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
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
    
    nBlocks = (nBlocks + blockSize - 1) / blockSize;
    Reduce<<<nBlocks, blockSize, blockSize * sizeof(float)>>>(d_mul, d_res);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float run_time_ms = 0;
    cudaEventElapsedTime(&run_time_ms, start, stop);
    std::cout << run_time_ms << std::endl;
    
    float res = 0;
    cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_mul);
    cudaFree(d_res);
    return res;
}

