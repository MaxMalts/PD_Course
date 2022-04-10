#pragma once

__global__ void MatrixVectorMul(int height, int width, int pitch, float* matrix, float* vector, float* result);
