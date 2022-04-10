#include <stdio.h>
#include <MatrixMul.cuh>

__global__
void MatrixMul(int heightA, int widthA, int widthB, float* matrixA, float* matrixB, float* matrixResult) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int rowStride = blockDim.x * gridDim.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int colStride = blockDim.y * gridDim.y;
        
    for (int i = row; i < heightA; i += rowStride) {
        for (int j = col; j < widthB; j += colStride) {
            matrixResult[i * widthB + j] = 0;
            for (int k = 0; k < widthA; ++k) {
                matrixResult[i * widthB + j] += matrixA[i * widthA + k] * matrixB[k * widthB + j];
            }
        }
    }
}