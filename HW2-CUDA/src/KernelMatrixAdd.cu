#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < pitch / sizeof(float) * height; i += stride) {
        result[i] = A[i] + B[i];
    }
}