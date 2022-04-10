#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, int pitch, float* matrix, float* vector, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < height; i += stride) {
        result[i] = 0;
        for (int j = 0; j < width; ++j) {
            result[i] += matrix[i * pitch / sizeof(float) + j] * vector[j];
        }
    }
}