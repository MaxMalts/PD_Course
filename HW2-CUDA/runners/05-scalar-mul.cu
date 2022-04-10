#include <iostream>
#include <ScalarMulRunner.cuh>

int main() {
    int nElements = 0, blockSize = 0;
    std::cin >> nElements >> blockSize;
    
    float* x = (float*)malloc(nElements * sizeof(float));
    float* y = (float*)malloc(nElements * sizeof(float));

    for(int i = 0; i < nElements; ++i) {
        x[i] = 2.0f;
        y[i] = 3.0f;
    }

    //float res1 = ScalarMulTwoReductions(nElements, x, y, blockSize);
    float res2 = ScalarMulSumPlusReduction(nElements, x, y, blockSize);

    free(x);
    free(y);
    return 0;
}